#pragma once

// ninfer::ops::detail - sm_89 INT8-family small-T launcher definition.
//
// Included by exactly one translation unit per KV storage; that unit explicitly instantiates the
// launcher with NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE(storage). The body, its route table and its
// launch bounds are unchanged from when small_t.cu held them inline.

#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

#include "core/device.h" // CUDA_CHECK
#include "ops/softmax_attention/dense/causal_cache/small_t_i8.cuh"

#include <cstdint>

#if defined(NINFER_SM89)

namespace ninfer::ops::detail {

// sm_89 INT8-family launcher. RotateK/RotateV select the rotated family's per-group H64 codecs;
// RotateK256 selects the D256-rotated Int8Group64 contract shared with the 120a build.
template <typename Geometry, int TokenTile, CausalSmallTI8Storage Storage, bool MultiBatch,
          bool Masked, typename CacheInput>
void causal_small_t_i8_launch(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                              PagedKVBatchLayerView cache,
                              const CausalSmallTInvocation& invocation,
                              std::int32_t logical_capacity, std::int32_t implementation_window,
                              std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                              Tensor& partial_l, cudaStream_t stream) {
    using Flags                      = CausalSmallTI8Flags<Storage>;
    constexpr bool PackedV           = Flags::PackedV;
    constexpr bool RotateK           = Flags::RotateK;
    constexpr bool RotateV           = Flags::RotateV;
    constexpr bool PackedK           = Flags::PackedK;
    constexpr bool E8Lattice         = Flags::E8Lattice;
    constexpr bool E8Root            = Flags::E8Root;
    constexpr bool RotateK256        = Flags::RotateK256;

    const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    const auto launch =
        [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kCausalHeadDim) : 0ULL;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                         MinBlocksPerSm, KeyBlock, DynamicArena,
                                                         PackedV, RotateK, RotateV, PackedK,
                                                         E8Lattice, E8Root, RotateK256, MultiBatch,
                                                         Masked, CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        causal_attention_small_t_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                                 KeyBlock, DynamicArena, PackedV, RotateK, RotateV,
                                                 PackedK, E8Lattice, E8Root, RotateK256,
                                                 MultiBatch, Masked, CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<float*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile >= 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiation of one storage's whole launcher set: both head geometries, every token
// width the dispatcher reaches for that geometry (T=7/8 are 24-head only), both batch modes, both
// mask modes and both cache inputs.
#define NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_ONE(GEOMETRY, TOKENS, STORAGE, INPUT)                 \
    template void causal_small_t_i8_launch<GEOMETRY, TOKENS, STORAGE, false, false, INPUT>(        \
        const Tensor&, INPUT, const Tensor&, float, PagedKVBatchLayerView,                         \
        const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, \
        Tensor&, cudaStream_t);                                                                    \
    template void causal_small_t_i8_launch<GEOMETRY, TOKENS, STORAGE, false, true, INPUT>(         \
        const Tensor&, INPUT, const Tensor&, float, PagedKVBatchLayerView,                         \
        const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, \
        Tensor&, cudaStream_t);                                                                    \
    template void causal_small_t_i8_launch<GEOMETRY, TOKENS, STORAGE, true, false, INPUT>(         \
        const Tensor&, INPUT, const Tensor&, float, PagedKVBatchLayerView,                         \
        const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, \
        Tensor&, cudaStream_t);                                                                    \
    template void causal_small_t_i8_launch<GEOMETRY, TOKENS, STORAGE, true, true, INPUT>(          \
        const Tensor&, INPUT, const Tensor&, float, PagedKVBatchLayerView,                         \
        const CausalSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&, \
        Tensor&, cudaStream_t);

#define NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(GEOMETRY, TOKENS, STORAGE)                     \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_ONE(GEOMETRY, TOKENS, STORAGE, CausalAppendInput)         \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_ONE(GEOMETRY, TOKENS, STORAGE, CausalCachedInput)

#define NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE(STORAGE)                                              \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 1, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 2, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 3, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 4, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 5, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 6, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 7, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H24Kv4, 8, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H16Kv2, 1, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H16Kv2, 2, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H16Kv2, 3, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H16Kv2, 4, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H16Kv2, 5, STORAGE)                      \
    NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE_TOKENS(CausalD256H16Kv2, 6, STORAGE)

} // namespace ninfer::ops::detail

#endif // NINFER_SM89
