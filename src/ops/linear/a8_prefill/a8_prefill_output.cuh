#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct A8PrefillIdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

// Plain Linear output contract: BF16 [N,T] with N stored fastest, i.e. out[token * rows + row].
struct A8PrefillContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store_vector(std::int32_t row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(data + static_cast<std::int64_t>(token) * rows + row, values);
    }
};

} // namespace ninfer::ops::detail
