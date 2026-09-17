#include "ops/linear/q5/q5_dispatch.h"
#include "ops/linear/q5/q5_shapes.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct ShapeEntry {
    std::int32_t n, k;
    Q5Launch (*select)(std::int32_t);
};

constexpr std::array kShapes{
    ShapeEntry{1024, 5120, select_q5_n1024_k5120},   ShapeEntry{6144, 5120, select_q5_n6144_k5120},
    ShapeEntry{7168, 5120, select_q5_n7168_k5120},   ShapeEntry{5120, 6144, select_q5_n5120_k6144},
    ShapeEntry{5120, 17408, select_q5_n5120_k17408}, ShapeEntry{1152, 1152, select_q5_n1152_k1152},
    ShapeEntry{1152, 4304, select_q5_n1152_k4304},
};

struct A8ShapeEntry {
    std::int32_t n, k;
    A8PrefillLaunch (*select)(std::int32_t);
};

// The prefill shapes of the 27B artifact that the FP8 prefill family claims: the attention and
// GDN input projections plus both residual/down projections.
constexpr std::array kA8PrefillShapes{
    A8ShapeEntry{6144, 5120, select_q5_a8_prefill_n6144_k5120},
    A8ShapeEntry{7168, 5120, select_q5_a8_prefill_n7168_k5120},
    A8ShapeEntry{5120, 6144, select_q5_a8_prefill_n5120_k6144},
    A8ShapeEntry{5120, 17408, select_q5_a8_prefill_n5120_k17408},
};
} // namespace

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) throw std::invalid_argument("q5 linear: T must be positive");
    for (const auto& entry : kShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    throw std::invalid_argument("q5 linear: unsupported shape");
}

Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q5 linear: unsupported policy");
    return select_q5_a16_launch(n, k, t);
}

A8PrefillLaunch select_q5_a8_prefill_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                            LinearPolicy policy) {
    if (t <= 0) throw std::invalid_argument("q5 linear: T must be positive");
    if (!valid_linear_policy(policy)) throw std::invalid_argument("q5 linear: unsupported policy");
    if (!a8_prefill_admits(policy, k, t)) return nullptr;
    for (const auto& entry : kA8PrefillShapes) {
        if (entry.n == n && entry.k == k) return entry.select(t);
    }
    return nullptr;
}

std::size_t q5_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                               std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q5 linear workspace: invalid token interval");
    }
    (void)select_q5_launch(n, k, min_tokens, policy);
    (void)select_q5_launch(n, k, max_tokens, policy);
    // Admission is monotone in T, so the top of the interval both decides the route and sizes it.
    return select_q5_a8_prefill_launch(n, k, max_tokens, policy) != nullptr
               ? a8_prefill_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void q5_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream) {
    const A8PrefillLaunch a8 = select_q5_a8_prefill_launch(weight.n, weight.k, x.ne[1], policy);
    if (a8 != nullptr) {
        if (workspace == nullptr) {
            throw std::invalid_argument("q5 A8 prefill linear requires caller workspace");
        }
        a8(x, weight, out, *workspace, stream);
        return;
    }
    select_q5_launch(weight.n, weight.k, x.ne[1], policy)(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
