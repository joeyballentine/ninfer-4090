#pragma once

// Private launch policy for the sm_89 FP8 prefill routes: materialize the E4M3 activation once,
// then run one MMA grid over (row tile, token tile).

#include "core/device.h"
#include "ops/linear/a8_prefill/a8_prefill_mma.cuh"
#include "ops/linear/a8_prefill/a8_prefill_output.cuh"
#include "ops/linear/a8_prefill/a8_prefill_plan.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// The BF16 prefill routes of these formats settle on a 64-row x 128-token tile with one 64-wide
// quant group per K step (Q4MmaR64C128Schedule and its Q5 twin). The A8 route keeps that tile and
// only swaps the operand roles, so the token axis becomes the MMA's m16 axis: four warps, each
// owning 64 tokens x 32 rows, i.e. the same 16 accumulator fragments per warp as the BF16 route.
template <int HighBytesPerGroup>
using A8PrefillR64C128 =
    A8PrefillSchedule<128, 64, 3, 2, 2, 2, HighBytesPerGroup, Cache::cg, Cache::cg>;

template <class Codec, class Schedule, class Epilogue, class Output,
          class RowPolicy = A8PrefillIdentityRows, bool PairRows = false>
void launch_a8_prefill_grid(const Weight& w, const A8PrefillWorkspace& scratch,
                            std::int32_t tokens, std::int32_t row_tiles, Epilogue epilogue,
                            Output output, RowPolicy row_policy, cudaStream_t stream) {
    const std::int32_t groups_per_row = w.padded_shape[1] / Schedule::kBlockK;
    const std::int32_t token_tiles =
        (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const dim3 grid(static_cast<unsigned>(row_tiles), static_cast<unsigned>(token_tiles), 1U);
    const bool full = (tokens % Schedule::kBlockTokens) == 0;

    if (full) {
        a8_prefill_mma_kernel<Codec, Schedule, true, Epilogue, Output, RowPolicy, PairRows>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.qhigh),
                static_cast<const std::uint8_t*>(w.scales), tokens, w.k, groups_per_row, epilogue,
                output, row_policy);
    } else {
        a8_prefill_mma_kernel<Codec, Schedule, false, Epilogue, Output, RowPolicy, PairRows>
            <<<grid, Schedule::kThreads, 0, stream>>>(
                scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(w.qdata),
                static_cast<const std::uint8_t*>(w.qhigh),
                static_cast<const std::uint8_t*>(w.scales), tokens, w.k, groups_per_row, epilogue,
                output, row_policy);
    }
    CUDA_CHECK(cudaGetLastError());
}

// The kernel walks groups_per_row K tiles and reads the activation plane at the same offsets, so
// a padded tail would read past the materialized activation. Every registered prefill K is a
// multiple of 128 and therefore unpadded; reject anything else instead of silently misreading.
inline void validate_a8_prefill_problem(const Weight& w, const Tensor& x, const Tensor& out,
                                        std::int32_t block_rows) {
    if (w.padded_shape[1] != w.k) {
        throw std::invalid_argument("A8 prefill: padded K must equal logical K");
    }
    if ((w.k % kA8PrefillGroupK) != 0 || (w.n % block_rows) != 0) {
        throw std::invalid_argument("A8 prefill: unsupported weight geometry");
    }
    if (x.ne[0] != w.k || out.ne[0] != w.n) {
        throw std::invalid_argument("A8 prefill: operand geometry mismatch");
    }
}

// Plain Linear entry shared by Q4 and Q5.
template <class Codec, class Schedule>
void launch_a8_prefill_linear(const Tensor& x, const Weight& w, Tensor& out,
                              WorkspaceArena& workspace, cudaStream_t stream) {
    validate_a8_prefill_problem(w, x, out, Schedule::kBlockRows);
    auto scope = workspace.scope();
    const A8PrefillWorkspace scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], w.k);
    launch_fp8_a8_quantize(x, w, scratch, stream);
    launch_a8_prefill_grid<Codec, Schedule, A8PrefillIdentityEpilogue, A8PrefillContiguousOutput>(
        w, scratch, x.ne[1], w.n / Schedule::kBlockRows, A8PrefillIdentityEpilogue{},
        A8PrefillContiguousOutput{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        A8PrefillIdentityRows{}, stream);
}

} // namespace ninfer::ops::detail
