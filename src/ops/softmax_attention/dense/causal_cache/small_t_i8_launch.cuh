#pragma once

// ninfer::ops::detail - sm_89 INT8-family small-T launcher declarations.
//
// The int8 family instantiates causal_attention_small_t_i8_tiled_kernel over five KV storages,
// every token width and both head geometries. Compiling all of them in small_t.cu made that one
// translation unit's ptxas step dominate the build, so each storage now owns a translation unit
// (small_t_i8_sm89.cu, small_t_rk4v4.cu, small_t_rk4v4_e8.cu, small_t_rk8v4.cu,
// small_t_rk2v4_e8.cu) that explicitly instantiates the launcher below. small_t.cu keeps only the
// dispatch plus the bf16 storages that every target compiles. This mirrors how the Blackwell
// storages are split (small_t_fp8.cu, small_t_k8v4.cu, small_t_nvfp4.cu).

#include "ops/softmax_attention/dense/causal_cache/launch.h"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cstdint>

#if defined(NINFER_SM89)

namespace ninfer::ops::detail {

// One enumerator per sm_89 INT8-family KvCacheStorage. Selecting the kernel flags from a single
// tag keeps the flag sets in one place; the dispatcher's route table stays per storage.
enum class CausalSmallTI8Storage {
    Int8Group64,                    // D256-rotated K and Q, unrotated V (120a-compatible codec)
    RotatedInt4KeyInt4ValueGroup64, // rk4v4: packed int4 K and V, per-group H64 rotations
    RK4V4E8,                        // rk4v4 with the E8 Conway-Sloane lattice K encode
    RotatedInt8KeyInt4ValueGroup64, // rk8v4: int8 rotated K, packed int4 rotated V
    RK2V4E8,                        // rk2v4-e8: E8 cylinder-root K code, packed int4 rotated V
};

// Kernel template flags per storage. These are exactly the argument lists small_t.cu passed
// inline before the split.
template <CausalSmallTI8Storage Storage>
struct CausalSmallTI8Flags;

template <>
struct CausalSmallTI8Flags<CausalSmallTI8Storage::Int8Group64> {
    static constexpr bool PackedV    = false;
    static constexpr bool RotateK    = false;
    static constexpr bool RotateV    = false;
    static constexpr bool PackedK    = false;
    static constexpr bool E8Lattice  = false;
    static constexpr bool E8Root     = false;
    static constexpr bool RotateK256 = true;
};

template <>
struct CausalSmallTI8Flags<CausalSmallTI8Storage::RotatedInt4KeyInt4ValueGroup64> {
    static constexpr bool PackedV    = true;
    static constexpr bool RotateK    = true;
    static constexpr bool RotateV    = true;
    static constexpr bool PackedK    = true;
    static constexpr bool E8Lattice  = false;
    static constexpr bool E8Root     = false;
    static constexpr bool RotateK256 = false;
};

template <>
struct CausalSmallTI8Flags<CausalSmallTI8Storage::RK4V4E8> {
    static constexpr bool PackedV    = true;
    static constexpr bool RotateK    = true;
    static constexpr bool RotateV    = true;
    static constexpr bool PackedK    = true;
    static constexpr bool E8Lattice  = true;
    static constexpr bool E8Root     = false;
    static constexpr bool RotateK256 = false;
};

template <>
struct CausalSmallTI8Flags<CausalSmallTI8Storage::RotatedInt8KeyInt4ValueGroup64> {
    static constexpr bool PackedV    = true;
    static constexpr bool RotateK    = true;
    static constexpr bool RotateV    = true;
    static constexpr bool PackedK    = false;
    static constexpr bool E8Lattice  = false;
    static constexpr bool E8Root     = false;
    static constexpr bool RotateK256 = false;
};

template <>
struct CausalSmallTI8Flags<CausalSmallTI8Storage::RK2V4E8> {
    static constexpr bool PackedV    = true;
    static constexpr bool RotateK    = true;
    static constexpr bool RotateV    = true;
    static constexpr bool PackedK    = false;
    static constexpr bool E8Lattice  = false;
    static constexpr bool E8Root     = true;
    static constexpr bool RotateK256 = false;
};

// Defined in small_t_i8_launch_impl.cuh and explicitly instantiated once per storage in that
// storage's translation unit. small_t.cu sees this declaration only, so it instantiates no
// int8 kernel.
template <typename Geometry, int TokenTile, CausalSmallTI8Storage Storage, bool MultiBatch,
          bool Masked, typename CacheInput>
void causal_small_t_i8_launch(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                              PagedKVBatchLayerView cache,
                              const CausalSmallTInvocation& invocation,
                              std::int32_t logical_capacity, std::int32_t implementation_window,
                              std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                              Tensor& partial_l, cudaStream_t stream);

} // namespace ninfer::ops::detail

#endif // NINFER_SM89
