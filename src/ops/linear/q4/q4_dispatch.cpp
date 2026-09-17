#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q4Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q4_n1024_k5120},
    ShapeEntry{4096, 5120, select_q4_n4096_k5120},
    ShapeEntry{5120, 6144, select_q4_n5120_k6144},
    ShapeEntry{6144, 5120, select_q4_n6144_k5120},
    ShapeEntry{7168, 5120, select_q4_n7168_k5120},
    ShapeEntry{34816, 5120, select_q4_n34816_k5120},
    ShapeEntry{131072, 5120, select_q4_n131072_k5120},
    ShapeEntry{131072, 2048, select_q4_n131072_k2048},
    ShapeEntry{3456, 1152, select_q4_n3456_k1152},
    ShapeEntry{4304, 1152, select_q4_n4304_k1152},
};

struct A8ShapeEntry {
    std::int32_t n, k;
    A8PrefillLaunch (*select)(std::int32_t);
};

// The prefill shapes of the 27B artifact that the FP8 prefill family claims: the attention and
// GDN input projections, the SwiGLU gate/up bank, and the K=6144 residual projection.
constexpr std::array kA8PrefillShapes{
    A8ShapeEntry{4096, 5120, select_q4_a8_prefill_n4096_k5120},
    A8ShapeEntry{5120, 6144, select_q4_a8_prefill_n5120_k6144},
    A8ShapeEntry{6144, 5120, select_q4_a8_prefill_n6144_k5120},
    A8ShapeEntry{7168, 5120, select_q4_a8_prefill_n7168_k5120},
    A8ShapeEntry{34816, 5120, select_q4_a8_prefill_n34816_k5120},
};
} // namespace

Q4Launch select_q4_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q4 linear: unsupported shape");
}

Q4Launch select_q4_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    return select_q4_a16_launch(n, k, t);
}

A8PrefillLaunch select_q4_a8_prefill_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                            LinearPolicy policy) {
    if (t <= 0) throw std::invalid_argument("q4 linear: T must be positive");
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q4 linear: unsupported policy");
    if (!a8_prefill_admits(policy, k, t)) return nullptr;
    for (const auto& entry : kA8PrefillShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    return nullptr;
}

std::size_t q4_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                               std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q4 linear workspace: invalid token interval");
    }
    (void)select_q4_launch(n, k, min_tokens, policy);
    (void)select_q4_launch(n, k, max_tokens, policy);
    // Admission is monotone in T, so the top of the interval both decides the route and sizes it.
    return select_q4_a8_prefill_launch(n, k, max_tokens, policy) != nullptr
               ? a8_prefill_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void q4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream) {
    const A8PrefillLaunch a8 =
        select_q4_a8_prefill_launch(weight.n, weight.k, x.ne[1], policy);
    if (a8 != nullptr) {
        if (workspace == nullptr) {
            throw std::invalid_argument("q4 A8 prefill linear requires caller workspace");
        }
        a8(x, weight, out, *workspace, stream);
        return;
    }
    select_q4_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
