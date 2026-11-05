#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

// Shared memory of the leader CTA of a split-K cluster, past the kernel's shared storage: the
// barrier the other slices complete their bytes on, then their FP32 partials: 4 rows of one thread
// are 16 contiguous bytes, and the 128 threads' 4-row groups are contiguous, so a warp stores 512 B at once
template <uint32_t BLOCK_M, uint32_t kNumSplits, uint32_t kNumThreads>
struct SplitKReduceScratch {
    struct Storage {
        uint64_t partials_full;
        uint64_t padding;
        float4 partials[kNumSplits - 1][BLOCK_M / 4][kNumThreads];
    };
    static constexpr uint32_t kNumRowsPerGroup = 4;

    // The 4-row groups that hold tokens of the block, the only ones exchanged
    CUTLASS_DEVICE static uint32_t get_num_valid_groups(const uint32_t& base_m_idx, const uint32_t& shape_m) {
        const uint32_t num_rows = cute::min(BLOCK_M, shape_m - base_m_idx);
        return (num_rows + kNumRowsPerGroup - 1) / kNumRowsPerGroup;
    }

    Storage* storage;

    CUTLASS_DEVICE explicit SplitKReduceScratch(void* ptr): storage(static_cast<Storage*>(ptr)) {}

    CUTLASS_DEVICE void init_barrier() const {
        const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(&storage->partials_full));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(addr));
    }

    CUTLASS_DEVICE static uint32_t to_cluster_addr(const void* ptr, const uint32_t& cta_rank) {
        uint32_t remote_addr;
        const auto local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(local_addr), "r"(cta_rank));
        return remote_addr;
    }
};

// Swap-AB split-K epilogue over a cluster of kNumSplits CTAs, one K slice each: TMEM lane `t` of the
// warpgroup holds output column `base_n_idx + t` and its accumulator columns the tokens of the block.
// The other slices store their FP32 partials into the leader's shared memory (st.async, completing
// bytes on the leader's barrier); the leader sums the slices in slice order and stores the BF16 tile
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSplits, uint32_t kNumUMMAStoreThreads,
          typename epilogue_op_t>
CUTLASS_DEVICE void
sm100_split_k_reduce_swap_ab(const SplitKReduceScratch<BLOCK_M, kNumSplits, kNumUMMAStoreThreads>& scratch,
                             const uint32_t& tmem_base_addr,
                             const uint32_t& base_m_idx, const uint32_t& base_n_idx,
                             const uint32_t& split_k_idx,
                             const uint32_t& shape_m, const uint32_t& shape_n,
                             const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                             const epilogue_op_t& epilogue_op,
                             const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier) {
    constexpr uint32_t kNumRowsPerLoad = 8;
    constexpr uint32_t kNumRowsPerStore = 4;
    DG_STATIC_ASSERT(BLOCK_N == kNumUMMAStoreThreads and kNumUMMAStoreThreads == 128, "One warpgroup per 128 columns");
    DG_STATIC_ASSERT(BLOCK_M % kNumRowsPerLoad == 0, "Invalid block M");

    // Accumulators into registers, then release the tensor memory
    uint32_t values[BLOCK_M];
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_M / kNumRowsPerLoad; ++ i) {
        cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_base_addr + i * kNumRowsPerLoad,
                                              values[i * 8 + 0], values[i * 8 + 1], values[i * 8 + 2], values[i * 8 + 3],
                                              values[i * 8 + 4], values[i * 8 + 5], values[i * 8 + 6], values[i * 8 + 7]);
    }
    cutlass::arch::fence_view_async_tmem_load();
    ptx::tcgen05_before_thread_sync();
    tmem_empty_barrier->arrive();

    using scratch_t = SplitKReduceScratch<BLOCK_M, kNumSplits, kNumUMMAStoreThreads>;
    const uint32_t thread_idx = epilogue_warp_idx * 32 + lane_idx;
    const uint32_t num_valid_groups = scratch_t::get_num_valid_groups(base_m_idx, shape_m);
    auto storage = scratch.storage;
    if (split_k_idx != 0) {
        // Send this slice's partial to the leader
        constexpr uint32_t kLeaderRank = 0;
        const auto remote_barrier = scratch.to_cluster_addr(&storage->partials_full, kLeaderRank);
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_M / kNumRowsPerStore; ++ i) {
            if (i >= num_valid_groups)
                break;
            const auto remote_data = scratch.to_cluster_addr(
                &storage->partials[split_k_idx - 1][i][thread_idx], kLeaderRank);
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                         :: "r"(remote_data), "r"(values[i * 4 + 0]), "r"(values[i * 4 + 1]),
                            "r"(values[i * 4 + 2]), "r"(values[i * 4 + 3]), "r"(remote_barrier) : "memory");
        }
        return;
    }

    // The leader waits for the other slices and sums in slice order
    const auto barrier_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&storage->partials_full));
    if (thread_idx == 0)
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(barrier_addr), "r"((kNumSplits - 1) * num_valid_groups * kNumUMMAStoreThreads * static_cast<uint32_t>(sizeof(float4))) : "memory");
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra LAB_WAIT_%=;\n"
        "}\n" :: "r"(barrier_addr), "n"(0) : "memory");
    float sums[BLOCK_M];
    #pragma unroll
    for (uint32_t row = 0; row < BLOCK_M; ++ row)
        sums[row] = __uint_as_float(values[row]);
    #pragma unroll
    for (uint32_t split = 1; split < kNumSplits; ++ split) {
        #pragma unroll
        for (uint32_t i = 0; i < BLOCK_M / kNumRowsPerStore; ++ i) {
            if (i >= num_valid_groups)
                break;
            const auto partial = storage->partials[split - 1][i][thread_idx];
            sums[i * 4 + 0] += partial.x;
            sums[i * 4 + 1] += partial.y;
            sums[i * 4 + 2] += partial.z;
            sums[i * 4 + 3] += partial.w;
        }
    }
    const uint32_t n_idx = base_n_idx + thread_idx;
    auto out = reinterpret_cast<nv_bfloat16*>(epilogue_op.act_out);
    #pragma unroll
    for (uint32_t row = 0; row < BLOCK_M; ++ row) {
        const uint32_t m_idx = base_m_idx + row;
        if (m_idx < shape_m and n_idx < shape_n)
            out[static_cast<uint64_t>(m_idx) * shape_n + n_idx] = __float2bfloat16_rn(sums[row]);
    }
}

} // namespace deep_gemm::epilogue
