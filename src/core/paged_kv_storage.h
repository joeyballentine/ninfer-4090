#pragma once

#include "core/dtype.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kD256KVCacheHeadDim = 256;

/** Physical data/scale planes for one K or V vector. */
struct PagedKVVectorLayout {
    DType data_dtype                  = DType::BF16;
    std::int32_t data_leading_extent  = 0;
    DType scale_dtype                 = DType::U8;
    std::int32_t scale_leading_extent = 0;

    [[nodiscard]] constexpr bool has_scale() const noexcept { return scale_leading_extent != 0; }

    [[nodiscard]] std::size_t physical_bytes() const {
        return static_cast<std::size_t>(data_leading_extent) * dtype_size(data_dtype) +
               static_cast<std::size_t>(scale_leading_extent) * dtype_size(scale_dtype);
    }

    friend bool operator==(const PagedKVVectorLayout&, const PagedKVVectorLayout&) = default;
};

/** Resolved physical plane schema for one paged K/V layer. */
struct PagedKVStorageLayout {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    std::int32_t head_dim  = 0;
    PagedKVVectorLayout key;
    PagedKVVectorLayout value;

    [[nodiscard]] constexpr std::size_t planes_per_layer() const noexcept {
        return 2ULL + static_cast<std::size_t>(key.has_scale()) +
               static_cast<std::size_t>(value.has_scale());
    }

    [[nodiscard]] constexpr std::size_t logical_vector_bytes() const noexcept {
        return static_cast<std::size_t>(head_dim) * sizeof(std::uint16_t);
    }

    [[nodiscard]] constexpr std::size_t logical_bytes_per_token_head() const noexcept {
        return 2ULL * logical_vector_bytes();
    }

    [[nodiscard]] std::size_t physical_bytes_per_token_head() const {
        return key.physical_bytes() + value.physical_bytes();
    }

    friend bool operator==(const PagedKVStorageLayout&, const PagedKVStorageLayout&) = default;
};

[[nodiscard]] inline PagedKVStorageLayout paged_kv_storage_layout(KvCacheStorage storage,
                                                                  std::int32_t head_dim) {
    if (head_dim <= 0) { throw std::invalid_argument("KV-cache head dimension must be positive"); }

    const auto symmetric = [=](PagedKVVectorLayout vector) {
        return PagedKVStorageLayout{storage, head_dim, vector, vector};
    };
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return {storage,
                head_dim,
                {DType::BF16, head_dim, DType::U8, 0},
                {DType::FP16, head_dim, DType::U8, 0}};
    case KvCacheStorage::Int8Group64:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::I8, 256, DType::FP16, 4}); }
        break;
    case KvCacheStorage::Fp8E4M3Row256:
        if (head_dim == kD256KVCacheHeadDim) {
            return symmetric({DType::FP8_E4M3FN, 256, DType::FP16, 1});
        }
        break;
    case KvCacheStorage::Nvfp4Group16:
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::U8, 128, DType::U8, 16}); }
        break;
    case KvCacheStorage::Fp8KeyNvfp4Value:
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::FP8_E4M3FN, 256, DType::FP16, 1},
                    {DType::U8, 128, DType::U8, 16}};
        }
        break;
    case KvCacheStorage::RotatedInt4KeyInt4ValueGroup64:
    case KvCacheStorage::RK4V4E8:
        // K y V: 256 dims int4 = 128 bytes U8 por vector; escala FP16 por grupo-64 (4/vector).
        // (La retícula E8 solo cambia cómo se eligen los códigos, no el layout físico.)
        if (head_dim == kD256KVCacheHeadDim) { return symmetric({DType::U8, 128, DType::FP16, 4}); }
        break;
    case KvCacheStorage::RK2V4E8:
        // K: 2 bytes (raiz, radio|eje) por cada 8 dims = 64 B + escala FP16 por grupo-64 (8 B).
        // V: 256 dims int4 = 128 B U8 + escala FP16 por grupo-64 (8 B). Total 208 B.
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::U8, 64, DType::FP16, 4},
                    {DType::U8, 128, DType::FP16, 4}};
        }
        break;
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        // K: 256 códigos int8 (256 B) + escala FP16 por grupo-64 (8 B) = 264 B.
        // V: 256 dims int4 = 128 B U8 + escala FP16 por grupo-64 (8 B) = 136 B. Total 400 B.
        if (head_dim == kD256KVCacheHeadDim) {
            return {storage,
                    head_dim,
                    {DType::I8, 256, DType::FP16, 4},
                    {DType::U8, 128, DType::FP16, 4}};
        }
        break;
    }
    throw std::invalid_argument("unsupported paged KV-cache storage geometry");
}

/**
 * Resolves one schedule into the physical plane schema of every KV-bearing layer.
 *
 * Layer `l` is the l-th full-attention layer of the text stack, not the l-th text layer: the
 * interleaved GDN layers hold a fixed recurrent state and never reach a paged pool. A uniform
 * schedule yields the same layout for every layer and reproduces single-kind planning exactly.
 */
[[nodiscard]] inline std::vector<PagedKVStorageLayout>
paged_kv_schedule_layouts(const KvCacheSchedule& schedule, std::uint32_t layers,
                          std::int32_t head_dim) {
    if (layers == 0) { throw std::invalid_argument("paged KV schedule needs at least one layer"); }
    if (!schedule.uniform() && schedule.head_layers >= layers) {
        throw std::invalid_argument(
            "KV-cache schedule boundary must be below the model's full-attention layer count");
    }
    std::vector<PagedKVStorageLayout> out;
    out.reserve(layers);
    const PagedKVStorageLayout head =
        schedule.uniform() ? PagedKVStorageLayout{}
                           : paged_kv_storage_layout(schedule.head, head_dim);
    const PagedKVStorageLayout tail = paged_kv_storage_layout(schedule.tail, head_dim);
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        out.push_back(layer < schedule.head_layers ? head : tail);
    }
    return out;
}

/** Physical bytes one token of one layer occupies across every KV head. */
[[nodiscard]] inline std::size_t paged_kv_layer_bytes_per_token(const PagedKVStorageLayout& layout,
                                                                std::int32_t num_kv_heads) {
    if (num_kv_heads <= 0) { throw std::invalid_argument("KV head count must be positive"); }
    return layout.physical_bytes_per_token_head() * static_cast<std::size_t>(num_kv_heads);
}

/** Physical bytes one token occupies across every layer of a scheduled pool. */
[[nodiscard]] inline std::size_t
paged_kv_bytes_per_token(std::span<const PagedKVStorageLayout> layers, std::int32_t num_kv_heads) {
    std::size_t total = 0;
    for (const PagedKVStorageLayout& layer : layers) {
        total += paged_kv_layer_bytes_per_token(layer, num_kv_heads);
    }
    return total;
}

} // namespace ninfer
