#pragma once

// Groupwise-integer weight x per-token-scaled E4M3 activation Tensor Core GEMM for Ada (sm_89).
//
// out[token, row] = a_row_scale[token] *
//                   sum_slabs w_group_scale[row, slab] * sum_{k in slab} a_code[token, k] *
//                                                        w_code[row, k]
//
// The MMA tile is oriented as [token,K] x [K,row], so the accumulator's contiguous axis is the
// public output-row axis, exactly like the row-scaled FP8 A8 kernel this body is modelled on.
// Two differences carry the groupwise weight format:
//
//   * the weight operand is staged as raw Q4/Q5 codes plus its binary16 group scales and decoded
//     in shared memory to *unscaled* E4M3 bytes.  Every code in [-16,15] is exactly representable
//     in E4M3, so that decode is lossless (see docs/maintainer/ada-fp8-prefill.md);
//   * one K tile is exactly one 64-wide quant group, i.e. two m16n8k32 MMA steps.  The group scale
//     differs between slabs, so each slab's partial product is scaled before it joins the running
//     FP32 total rather than at the end.  The per-token activation scale is constant in K and is
//     therefore applied once, in the epilogue.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One K tile is one quant group of the row-split G64 formats.
inline constexpr int kA8PrefillGroupK = 64;

template <int BlockTokens, int BlockRows, int Stages, int WarpsTokens, int WarpsRows,
          int MinBlocksPerSm, int HighBytesPerGroup, Cache WeightCache, Cache ActivationCache>
struct A8PrefillSchedule {
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockK            = kA8PrefillGroupK;
    static constexpr int kStages            = Stages;
    static constexpr int kWarpsTokens       = WarpsTokens;
    static constexpr int kWarpsRows         = WarpsRows;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kHighBytes         = HighBytesPerGroup;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;

    static constexpr int kWarps          = kWarpsTokens * kWarpsRows;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kWarpTokens     = kBlockTokens / kWarpsTokens;
    static constexpr int kWarpRows       = kBlockRows / kWarpsRows;
    static constexpr int kMmaTokens      = kWarpTokens / 16;
    static constexpr int kMmaRows        = kWarpRows / 8;
    static constexpr int kMmaK           = kBlockK / 32; // two k32 steps per 64-wide slab
    static constexpr int kSegmentsPerRow = kBlockK / 16;
    static constexpr int kCodeBytes      = 32; // one G64 group of 4/5-bit codes

    static constexpr int kActivationBytes = kStages * kBlockTokens * kBlockK;
    static constexpr int kCodePlaneBytes  = kStages * kBlockRows * kCodeBytes;
    static constexpr int kHighPlaneBytes  = kStages * kBlockRows * kHighBytes;
    // Rounded so the decoded-weight plane that follows keeps its 16-byte alignment.
    static constexpr int kScalePlaneBytes = ((kStages * kBlockRows * 2) + 15) & ~15;
    static constexpr int kWeightPlaneBytes = kBlockRows * kBlockK;
    static constexpr int kStagingBytes     = kActivationBytes + kCodePlaneBytes + kHighPlaneBytes +
                                         kScalePlaneBytes + kWeightPlaneBytes;
    // The BF16 output tile aliases the staging planes after the contraction.
    static constexpr int kOutputStride = kBlockRows + 8;
    static constexpr int kOutputBytes =
        kBlockTokens * kOutputStride * static_cast<int>(sizeof(__nv_bfloat16));
    static constexpr int kSharedBytes =
        kStagingBytes > kOutputBytes ? kStagingBytes : kOutputBytes;

    static_assert(kBlockTokens > 0 && kBlockRows > 0);
    static_assert((kBlockTokens % kWarpsTokens) == 0 && (kBlockRows % kWarpsRows) == 0);
    static_assert((kWarpTokens % 16) == 0 && (kWarpRows % 8) == 0);
    static_assert((kBlockRows % 8) == 0, "the vectorized output store writes eight rows at a time");
    static_assert(kStages >= 2 && kStages <= 8, "cp.async pipeline depth must fit cp_wait");
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert((kSegmentsPerRow & (kSegmentsPerRow - 1)) == 0);
    static_assert(kHighBytes == 0 || kHighBytes == 8, "only the Q5 high-bit plane is supported");
    static_assert(kSharedBytes <= 48 * 1024,
                  "A8 prefill staging exceeds the static 48 KiB sm_89 block budget; an opt-in "
                  "dynamic allocation would be required above it");
};

// XOR swizzle over the 16-byte segments of a [rows][kBlockK] byte tile, matching the swizzle the
// row-scaled FP8 kernel uses, so ldmatrix reads spread across shared-memory bank groups.
template <class Schedule>
__device__ __forceinline__ int a8_prefill_shared_byte(int row, int logical_byte) {
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & (Schedule::kSegmentsPerRow - 1));
    return physical_segment * 16 + byte_in_segment;
}

struct A8PrefillIdentityRows {
    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

template <class Codec, class Schedule, bool FullTokens, class Epilogue, class Output,
          class RowPolicy = A8PrefillIdentityRows, bool PairRows = false>
__global__ __launch_bounds__(Schedule::kThreads,
                             Schedule::kMinBlocksPerSm) void a8_prefill_mma_kernel(
    const std::uint8_t* __restrict__ activation_codes, const float* __restrict__ activation_scales,
    const std::uint8_t* __restrict__ weight_codes, const std::uint8_t* __restrict__ weight_high,
    const std::uint8_t* __restrict__ weight_scales, std::int32_t tokens, std::int32_t input_rows,
    std::int32_t groups_per_row, Epilogue epilogue, Output output, RowPolicy row_policy = {}) {
    constexpr int BM      = Schedule::kBlockTokens;
    constexpr int BN      = Schedule::kBlockRows;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int S       = Schedule::kStages;
    constexpr int THREADS = Schedule::kThreads;
    constexpr int MT      = Schedule::kMmaTokens;
    constexpr int MR      = Schedule::kMmaRows;
    constexpr int CB      = Schedule::kCodeBytes;
    constexpr int HB      = Schedule::kHighBytes;
    static_assert(Codec::kHighBytesPerGroup == HB, "codec and schedule disagree about Q5 high bits");
    static_assert(!PairRows || (BN % 2) == 0);

    __shared__ __align__(16) unsigned char shared_raw[Schedule::kSharedBytes];
    auto* activation_shared = shared_raw;
    auto* code_shared       = activation_shared + Schedule::kActivationBytes;
    auto* high_shared       = code_shared + Schedule::kCodePlaneBytes;
    auto* scale_shared      = high_shared + Schedule::kHighPlaneBytes;
    auto* weight_shared     = scale_shared + Schedule::kScalePlaneBytes;

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int warp_token = warp / Schedule::kWarpsRows;
    const int warp_row   = warp - warp_token * Schedule::kWarpsRows;

    constexpr int rows_per_block = PairRows ? BN / 2 : BN;
    const int row_begin          = static_cast<int>(blockIdx.x) * rows_per_block;
    const int token_begin        = static_cast<int>(blockIdx.y) * BM;
    const int k_tiles            = groups_per_row;

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k_begin      = k_tile * BK;
        auto* activation_stage = activation_shared + stage * BM * BK;

#pragma unroll 1
        for (int task = tid; task < BM * Schedule::kSegmentsPerRow; task += THREADS) {
            const int row             = task / Schedule::kSegmentsPerRow;
            const int logical_byte    = (task - row * Schedule::kSegmentsPerRow) * 16;
            const int physical_byte   = a8_prefill_shared_byte<Schedule>(row, logical_byte);
            auto* destination         = activation_stage + row * BK + physical_byte;
            const int token           = token_begin + row;
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    destination, activation_codes + static_cast<std::int64_t>(token) * input_rows +
                                     k_begin + logical_byte);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    activation_codes + static_cast<std::int64_t>(valid ? token : 0) * input_rows +
                        k_begin + logical_byte,
                    valid ? 16 : 0);
            }
        }

        // Raw weight codes: one 32-byte group per staged row, issued as two 16-byte copies.
#pragma unroll 1
        for (int task = tid; task < BN * (CB / 16); task += THREADS) {
            const int row          = task / (CB / 16);
            const int chunk        = task - row * (CB / 16);
            const int weight_row   = row_policy.weight_row(row_begin, row);
            const std::int64_t idx = static_cast<std::int64_t>(weight_row) * groups_per_row + k_tile;
            cp_async<16, Schedule::kWeightCache>(code_shared + stage * BN * CB + row * CB +
                                                     chunk * 16,
                                                 weight_codes + idx * CB + chunk * 16);
        }

        if constexpr (HB > 0) {
#pragma unroll 1
            for (int row = tid; row < BN; row += THREADS) {
                const int weight_row = row_policy.weight_row(row_begin, row);
                const std::int64_t idx =
                    static_cast<std::int64_t>(weight_row) * groups_per_row + k_tile;
                // cp.async.cg is 16-byte only; the eight-byte high plane uses the .ca form.
                cp_async<8>(high_shared + stage * BN * HB + row * HB, weight_high + idx * HB);
            }
        }

        // One binary16 group scale per staged row. The stride between a row's consecutive groups
        // is groups_per_row, so this plane is gathered rather than copied asynchronously.
#pragma unroll 1
        for (int row = tid; row < BN; row += THREADS) {
            const int weight_row = row_policy.weight_row(row_begin, row);
            const std::int64_t idx = static_cast<std::int64_t>(weight_row) * groups_per_row + k_tile;
            reinterpret_cast<std::uint16_t*>(scale_shared + stage * BN * 2)[row] =
                *reinterpret_cast<const std::uint16_t*>(weight_scales + idx * 2);
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    float accumulators[MT][MR][4] = {};
    const int a_matrix            = lane >> 3;
    const int a_row_offset        = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte       = (a_matrix >> 1) * 16;
    const int b_row_offset        = lane & 7;
    const int b_column_byte       = ((lane >> 3) & 1) * 16;
    const int accumulator_row     = 2 * (lane & 3);

#pragma unroll 1
    for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= k_tiles) {
            cp_wait<S - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        // Decode the staged group to unscaled E4M3 bytes, four logical K values per store.
        const auto* code_stage = code_shared + stage * BN * CB;
        const auto* high_stage = high_shared + stage * BN * HB;
#pragma unroll 1
        for (int task = tid; task < BN * (BK / 4); task += THREADS) {
            const int row  = task / (BK / 4);
            const int word = task - row * (BK / 4);
            const unsigned value =
                Codec::decode_word(code_stage + row * CB, high_stage + row * HB, word);
            const int physical_byte = a8_prefill_shared_byte<Schedule>(row, word * 4);
            *reinterpret_cast<unsigned*>(weight_shared + row * BK + physical_byte) = value;
        }

        // This slab's weight group scales, for the two output rows this lane owns per row tile.
        float weight_scale[MR][2];
#pragma unroll
        for (int mma_row = 0; mma_row < MR; ++mma_row) {
            const int local_row = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
            const std::uint32_t bits = *reinterpret_cast<const std::uint32_t*>(
                scale_shared + stage * BN * 2 + local_row * 2);
            const float2 pair       = __half22float2(half2_from_bits(bits));
            weight_scale[mma_row][0] = pair.x;
            weight_scale[mma_row][1] = pair.y;
        }
        __syncthreads();

        unsigned a_fragments[Schedule::kMmaK][MT][4];
        unsigned b_fragments[Schedule::kMmaK][MR][2];
#pragma unroll
        for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
#pragma unroll
            for (int mma_token = 0; mma_token < MT; ++mma_token) {
                const int row = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
                const int physical_byte =
                    a8_prefill_shared_byte<Schedule>(row, k_step * 32 + a_column_byte);
                ldmatrix_x4(a_fragments[k_step][mma_token][0], a_fragments[k_step][mma_token][1],
                            a_fragments[k_step][mma_token][2], a_fragments[k_step][mma_token][3],
                            smem_addr(activation_shared + stage * BM * BK + row * BK +
                                      physical_byte));
            }
#pragma unroll
            for (int mma_row = 0; mma_row < MR; ++mma_row) {
                const int row = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
                const int physical_byte =
                    a8_prefill_shared_byte<Schedule>(row, k_step * 32 + b_column_byte);
                ldmatrix_x2(b_fragments[k_step][mma_row][0], b_fragments[k_step][mma_row][1],
                            smem_addr(weight_shared + row * BK + physical_byte));
            }
        }

        // One 64-wide slab at a time: accumulate both k32 steps into a private partial, then fold
        // this slab's weight group scale in before adding it to the running total.
#pragma unroll
        for (int mma_token = 0; mma_token < MT; ++mma_token) {
#pragma unroll
            for (int mma_row = 0; mma_row < MR; ++mma_row) {
                float partial[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
                for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
                    mma_fp8_e4m3(partial[0], partial[1], partial[2], partial[3],
                                 a_fragments[k_step][mma_token][0],
                                 a_fragments[k_step][mma_token][1],
                                 a_fragments[k_step][mma_token][2],
                                 a_fragments[k_step][mma_token][3],
                                 b_fragments[k_step][mma_row][0], b_fragments[k_step][mma_row][1]);
                }
                accumulators[mma_token][mma_row][0] += partial[0] * weight_scale[mma_row][0];
                accumulators[mma_token][mma_row][1] += partial[1] * weight_scale[mma_row][1];
                accumulators[mma_token][mma_row][2] += partial[2] * weight_scale[mma_row][0];
                accumulators[mma_token][mma_row][3] += partial[3] * weight_scale[mma_row][1];
            }
        }

        __syncthreads();
        const int next_k_tile = k_tile + S;
        if (next_k_tile < k_tiles) {
            stage_inputs(stage, next_k_tile);
            cp_commit();
        }
    }

    const int accumulator_token = lane >> 2;
    auto* shared_output         = reinterpret_cast<__nv_bfloat16*>(shared_raw);
#pragma unroll
    for (int mma_token = 0; mma_token < MT; ++mma_token) {
        const int token0 =
            token_begin + warp_token * Schedule::kWarpTokens + mma_token * 16 + accumulator_token;
        const int token1 = token0 + 8;
        const float activation_scale0 =
            (FullTokens || token0 < tokens) ? activation_scales[token0] : 0.0F;
        const float activation_scale1 =
            (FullTokens || token1 < tokens) ? activation_scales[token1] : 0.0F;
#pragma unroll
        for (int mma_row = 0; mma_row < MR; ++mma_row) {
            const int local_row0  = warp_row * Schedule::kWarpRows + mma_row * 8 + accumulator_row;
            const int parent_row0 = row_policy.weight_row(row_begin, local_row0);
            const int parent_row1 = row_policy.weight_row(row_begin, local_row0 + 1);
            const float* values   = accumulators[mma_token][mma_row];
            float value00         = values[0] * activation_scale0;
            float value01         = values[1] * activation_scale0;
            float value10         = values[2] * activation_scale1;
            float value11         = values[3] * activation_scale1;
            if constexpr (FullTokens) {
                value00 = epilogue.apply(parent_row0, token0, value00);
                value01 = epilogue.apply(parent_row1, token0, value01);
                value10 = epilogue.apply(parent_row0, token1, value10);
                value11 = epilogue.apply(parent_row1, token1, value11);
            } else {
                if (token0 < tokens) {
                    value00 = epilogue.apply(parent_row0, token0, value00);
                    value01 = epilogue.apply(parent_row1, token0, value01);
                }
                if (token1 < tokens) {
                    value10 = epilogue.apply(parent_row0, token1, value10);
                    value11 = epilogue.apply(parent_row1, token1, value11);
                }
            }
            auto* destination0 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + (token0 - token_begin) * Schedule::kOutputStride + local_row0);
            auto* destination1 = reinterpret_cast<__nv_bfloat162*>(
                shared_output + (token1 - token_begin) * Schedule::kOutputStride + local_row0);
            *destination0 = __floats2bfloat162_rn(value00, value01);
            *destination1 = __floats2bfloat162_rn(value10, value11);
        }
    }
    __syncthreads();

    constexpr int stored_rows       = PairRows ? BN / 2 : BN;
    constexpr int vectors_per_token = stored_rows / 8;
    constexpr int output_vectors    = BM * vectors_per_token;
#pragma unroll 1
    for (int task = tid; task < output_vectors; task += THREADS) {
        const int token_local = task / vectors_per_token;
        const int row_vector  = task - token_local * vectors_per_token;
        const int token       = token_begin + token_local;
        if (!FullTokens && token >= tokens) { continue; }
        const uint4 values = load_vec<uint4>(shared_output + token_local * Schedule::kOutputStride +
                                             row_vector * 8);
        if constexpr (PairRows) {
            const uint4 paired =
                load_vec<uint4>(shared_output + token_local * Schedule::kOutputStride +
                                stored_rows + row_vector * 8);
            output.store_pair_vector(row_begin + row_vector * 8, token, values, paired);
        } else {
            output.store_vector(row_begin + row_vector * 8, token, values);
        }
    }
}

} // namespace ninfer::ops::detail
