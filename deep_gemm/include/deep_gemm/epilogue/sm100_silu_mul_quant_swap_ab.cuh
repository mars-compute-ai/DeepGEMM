#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace deep_gemm::epilogue {

/// Scratch of the fused SiLU-and-mul quant epilogue: the token exchange lies over the unused C/D staging
/// buffer, the amax reduction and the barriers in `kNumExtraSmemBytes` past the shared storage
template <uint32_t kNumTokensPerCTA>
struct SiluMulQuantScratch {
    static constexpr uint32_t kNumThreads = 128;
    static constexpr uint32_t kNumSlots = 2;
    static constexpr uint32_t kNumPackedPerThread = kNumTokensPerCTA * sizeof(nv_bfloat16) / sizeof(uint4);
    static constexpr uint32_t kNumExtraSmemBytes = 1024;
    DG_STATIC_ASSERT(kNumTokensPerCTA == 8 or kNumTokensPerCTA == 16, "Invalid tokens per CTA");

    struct Exchange {
        // Each thread receives the peer's tokens of its row as BF16
        uint4 data[kNumSlots][kNumThreads][kNumPackedPerThread];
    };
    struct Extra {
        // Per-warp amax of this CTA's tokens, as BF16 pairs
        uint4 warp_amax[kNumSlots][kNumThreads / 32][kNumPackedPerThread];
        // Peer data landed, one barrier per slot, expecting every peer epilogue thread
        uint64_t exchange_full[kNumSlots];
    };
    DG_STATIC_ASSERT(sizeof(Extra) <= kNumExtraSmemBytes, "Fused scratch exceeds its shared memory");

    Exchange* exchange;
    Extra* extra;

    CUTLASS_DEVICE SiluMulQuantScratch(void* exchange_ptr, void* extra_ptr):
        exchange(static_cast<Exchange*>(exchange_ptr)), extra(static_cast<Extra*>(extra_ptr)) {}

    CUTLASS_DEVICE static uint32_t to_cluster_addr(const void* ptr, const uint32_t& cta_rank) {
        uint32_t remote_addr;
        const auto local_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(local_addr), "r"(cta_rank));
        return remote_addr;
    }

    CUTLASS_DEVICE void init_barriers() const {
        #pragma unroll
        for (uint32_t i = 0; i < kNumSlots; ++ i) {
            const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(&extra->exchange_full[i]));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(kNumThreads));
        }
    }

    /// Store this thread's tokens into the peer's slot and arrive at the peer's barrier
    CUTLASS_DEVICE void send(const uint32_t& slot, const uint32_t& thread_idx, const uint32_t& peer_rank,
                             const uint4 (&values)[kNumPackedPerThread]) const {
        #pragma unroll
        for (uint32_t i = 0; i < kNumPackedPerThread; ++ i) {
            const auto remote_data = to_cluster_addr(&exchange->data[slot][thread_idx][i], peer_rank);
            asm volatile("st.shared::cluster.v4.b32 [%0], {%1, %2, %3, %4};"
                         :: "r"(remote_data), "r"(values[i].x), "r"(values[i].y), "r"(values[i].z), "r"(values[i].w) : "memory");
        }
        const auto remote_barrier = to_cluster_addr(&extra->exchange_full[slot], peer_rank);
        asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(remote_barrier) : "memory");
    }

    /// Wait for the peer's tokens of this thread's row and read them
    CUTLASS_DEVICE void receive(const uint32_t& slot, const uint32_t& phase, const uint32_t& thread_idx,
                                uint4 (&values)[kNumPackedPerThread]) const {
        const auto addr = static_cast<uint32_t>(__cvta_generic_to_shared(&extra->exchange_full[slot]));
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "LAB_WAIT_%=:\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;\n"
            "@!p bra LAB_WAIT_%=;\n"
            "}\n" :: "r"(addr), "r"(phase) : "memory");
        #pragma unroll
        for (uint32_t i = 0; i < kNumPackedPerThread; ++ i)
            values[i] = exchange->data[slot][thread_idx][i];
    }
};

/// The masked SiLU-and-mul quant's SiLU on BF16-rounded gates (Blackwell form)
CUTLASS_DEVICE float silu_mul_quant_silu(const float& value) {
    const float half = 0.5f * value;
    return half * (1.0f + __tanhf(half));
}

/// Fused SiLU-and-mul + per-token-group FP8 quant of one swap-AB tile of a 2-CTA cluster:
/// CTA 0 holds the gate rows and CTA 1 the up rows of the same 128 activation columns for BLOCK_M tokens;
/// CTA 0 finishes the first half of the tokens and CTA 1 the second, after a DSMEM exchange of the other half
template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kActShapeN, typename scratch_t, typename epilogue_op_t>
CUTLASS_DEVICE void
sm100_silu_mul_quant_swap_ab(const scratch_t& scratch,
                             const uint32_t& tile_idx,
                             const uint32_t& tmem_base_addr,
                             const uint32_t& group_idx, const uint32_t& m_block_idx,
                             const uint32_t& act_block_idx,
                             const uint32_t& shape_m, const uint32_t& masked_m,
                             const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx,
                             const epilogue_op_t& epilogue_op,
                             const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier) {
    constexpr uint32_t kNumTokensPerCTA = BLOCK_M / 2;
    constexpr uint32_t kNumPairs = kNumTokensPerCTA / 2;
    constexpr uint32_t kNumPacked = scratch_t::kNumPackedPerThread;
    constexpr uint32_t kGroupSize = transform::EpilogueSiluMulQuantFP8::kGroupSize;
    constexpr uint32_t kNumPackedSF = kActShapeN / kGroupSize / 4;
    constexpr float kFP8Max = 448.0f;
    constexpr float kFP8MaxInv = 1.0f / kFP8Max;
    constexpr float kAmaxFloor = 1e-10f;
    DG_STATIC_ASSERT((BLOCK_M == 16 or BLOCK_M == 32) and BLOCK_N == kGroupSize, "Invalid fused block");
    DG_STATIC_ASSERT(kNumPacked * 4 == kNumPairs, "Invalid packing");
    DG_STATIC_ASSERT(kActShapeN % (kGroupSize * 4) == 0, "SF words must be whole");

    const uint32_t cta_rank = cute::block_rank_in_cluster();
    const uint32_t peer_rank = cta_rank ^ 1;
    const uint32_t thread_idx = epilogue_warp_idx * 32 + lane_idx;
    const uint32_t slot = tile_idx % scratch_t::kNumSlots;
    const uint32_t phase = (tile_idx / scratch_t::kNumSlots) & 1;

    // Read this thread's row for the BLOCK_M tokens, then release the accumulator
    uint32_t values[BLOCK_M];
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_M / 16; ++ i) {
        cute::SM100_TMEM_LOAD_32dp32b16x::copy(tmem_base_addr + i * 16,
            values[i * 16 + 0], values[i * 16 + 1], values[i * 16 + 2], values[i * 16 + 3],
            values[i * 16 + 4], values[i * 16 + 5], values[i * 16 + 6], values[i * 16 + 7],
            values[i * 16 + 8], values[i * 16 + 9], values[i * 16 + 10], values[i * 16 + 11],
            values[i * 16 + 12], values[i * 16 + 13], values[i * 16 + 14], values[i * 16 + 15]);
    }
    cutlass::arch::fence_view_async_tmem_load();
    ptx::tcgen05_before_thread_sync();
    tmem_empty_barrier->arrive(0u);

    // Round into BF16 token pairs, as the BF16 D store does
    uint32_t packed[BLOCK_M / 2];
    #pragma unroll
    for (uint32_t i = 0; i < BLOCK_M / 2; ++ i)
        packed[i] = static_cast<uint32_t>(math::cast_into_bf16_and_pack(values[i * 2], values[i * 2 + 1]));

    // Keep this CTA's tokens and hand the other half to the peer
    const uint32_t keep_base = cta_rank * kNumPairs;
    const uint32_t send_base = peer_rank * kNumPairs;
    uint4 list_send[kNumPacked], list_received[kNumPacked];
    #pragma unroll
    for (uint32_t i = 0; i < kNumPacked; ++ i)
        list_send[i] = make_uint4(packed[send_base + i * 4], packed[send_base + i * 4 + 1],
                                  packed[send_base + i * 4 + 2], packed[send_base + i * 4 + 3]);
    scratch.send(slot, thread_idx, peer_rank, list_send);
    scratch.receive(slot, phase, thread_idx, list_received);
    const auto received = reinterpret_cast<const uint32_t*>(list_received);

    // SiLU(gate) in BF16, times up in BF16
    nv_bfloat162 act[kNumPairs];
    #pragma unroll
    for (uint32_t i = 0; i < kNumPairs; ++ i) {
        const uint32_t own = packed[keep_base + i];
        const uint32_t gate_bits = cta_rank == 0 ? own : received[i];
        const uint32_t up_bits = cta_rank == 0 ? received[i] : own;
        const auto gate = __bfloat1622float2(*reinterpret_cast<const nv_bfloat162*>(&gate_bits));
        const auto silu = __float22bfloat162_rn(make_float2(silu_mul_quant_silu(gate.x), silu_mul_quant_silu(gate.y)));
        act[i] = __hmul2(silu, *reinterpret_cast<const nv_bfloat162*>(&up_bits));
    }

    // Amax per token over the 128 columns: lanes, then warps
    nv_bfloat162 amax[kNumPairs];
    #pragma unroll
    for (uint32_t i = 0; i < kNumPairs; ++ i) {
        amax[i] = __habs2(act[i]);
        #pragma unroll
        for (uint32_t offset = 16; offset > 0; offset /= 2) {
            const uint32_t bits = __shfl_xor_sync(0xffffffffu, *reinterpret_cast<const uint32_t*>(&amax[i]), offset);
            amax[i] = __hmax2(amax[i], *reinterpret_cast<const nv_bfloat162*>(&bits));
        }
    }
    if (lane_idx == 0) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumPacked; ++ i)
            scratch.extra->warp_amax[slot][epilogue_warp_idx][i] = reinterpret_cast<const uint4*>(amax)[i];
    }
    cutlass::arch::NamedBarrier::sync(scratch_t::kNumThreads, 0);
    #pragma unroll
    for (uint32_t w = 0; w < scratch_t::kNumThreads / 32; ++ w) {
        uint4 list_warp_amax[kNumPacked];
        #pragma unroll
        for (uint32_t i = 0; i < kNumPacked; ++ i)
            list_warp_amax[i] = scratch.extra->warp_amax[slot][w][i];
        const auto warp_amax = reinterpret_cast<const nv_bfloat162*>(list_warp_amax);
        #pragma unroll
        for (uint32_t i = 0; i < kNumPairs; ++ i)
            amax[i] = __hmax2(amax[i], warp_amax[i]);
    }

    // UE8M0 SFs and the FP8 cast, per token
    const uint32_t token_base = m_block_idx * BLOCK_M + cta_rank * kNumTokensPerCTA;
    const auto act_out = static_cast<uint8_t*>(epilogue_op.act_out);
    const auto sf_out = reinterpret_cast<uint8_t*>(epilogue_op.sfd);
    const uint64_t sf_group_stride = static_cast<uint64_t>(epilogue_op.sfd_stride) * sizeof(uint32_t);
    const uint64_t sf_base = static_cast<uint64_t>(group_idx) * kNumPackedSF * sf_group_stride +
                             (act_block_idx / 4) * sf_group_stride + act_block_idx % 4;
    const uint64_t out_base = static_cast<uint64_t>(group_idx) * shape_m * kActShapeN +
                              act_block_idx * kGroupSize + thread_idx;
    const auto max_clip = __float2bfloat16_rn(kFP8Max);
    #pragma unroll
    for (uint32_t i = 0; i < kNumPairs; ++ i) {
        const auto amax_f32 = __bfloat1622float2(amax[i]);
        const auto raw_scale_x = fmaxf(amax_f32.x, kAmaxFloor) * kFP8MaxInv;
        const auto raw_scale_y = fmaxf(amax_f32.y, kAmaxFloor) * kFP8MaxInv;
        const uint32_t bits_x = __float_as_uint(raw_scale_x), bits_y = __float_as_uint(raw_scale_y);
        const uint32_t exp_x = ((bits_x >> 23) & 0xffu) + ((bits_x & 0x7fffffu) != 0);
        const uint32_t exp_y = ((bits_y >> 23) & 0xffu) + ((bits_y & 0x7fffffu) != 0);
        const auto quant_scale = __float22bfloat162_rn(make_float2(
            __uint_as_float((127u + 127u - exp_x) << 23), __uint_as_float((127u + 127u - exp_y) << 23)));
        const auto fp8x2 = __nv_fp8x2_e4m3(__hmin2_nan(__hmul2(act[i], quant_scale), nv_bfloat162(max_clip, max_clip)));
        const auto fp8x2_bits = static_cast<uint32_t>(fp8x2.__x);

        #pragma unroll
        for (uint32_t k = 0; k < 2; ++ k) {
            const uint32_t token_idx = token_base + i * 2 + k;
            if (token_idx < masked_m) {
                act_out[out_base + static_cast<uint64_t>(token_idx) * kActShapeN] = static_cast<uint8_t>(fp8x2_bits >> (k * 8));
                if (thread_idx == i * 2 + k)
                    sf_out[sf_base + token_idx * sizeof(uint32_t)] = static_cast<uint8_t>(k == 0 ? exp_x : exp_y);
            }
        }
    }
}

} // namespace deep_gemm::epilogue
