#pragma once

#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/a8_prefill/a8_prefill_plan.h"
#include "ops/linear/q5/q5_launch.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

// Null when the sm_89 FP8 prefill family does not claim this problem at this extent.
A8PrefillLaunch select_q5_a8_prefill_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                            LinearPolicy policy);

[[nodiscard]] std::size_t q5_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
                                                             LinearPolicy policy,
                                                             std::int32_t min_tokens,
                                                             std::int32_t max_tokens);

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
