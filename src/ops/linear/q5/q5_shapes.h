#pragma once

#include "ops/linear/a8_prefill/a8_prefill_plan.h"
#include "ops/linear/q5/q5_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Q5Launch select_q5_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n1152_k1152(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n1152_k4304(std::int32_t tokens);

// sm_89 FP8 prefill routes; null off sm_89, below the family threshold, or when the
// EngineOptions/CLI gate is off.
[[nodiscard]] A8PrefillLaunch select_q5_a8_prefill_n6144_k5120(std::int32_t tokens);
[[nodiscard]] A8PrefillLaunch select_q5_a8_prefill_n7168_k5120(std::int32_t tokens);
[[nodiscard]] A8PrefillLaunch select_q5_a8_prefill_n5120_k6144(std::int32_t tokens);
[[nodiscard]] A8PrefillLaunch select_q5_a8_prefill_n5120_k17408(std::int32_t tokens);

} // namespace ninfer::ops::detail
