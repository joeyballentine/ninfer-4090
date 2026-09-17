#pragma once

#include "ops/linear/a8_prefill/a8_prefill_plan.h"
#include "ops/linear/q4/q4_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Q4Launch select_q4_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4096_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k5120(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n131072_k2048(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n3456_k1152(std::int32_t tokens);
[[nodiscard]] Q4Launch select_q4_n4304_k1152(std::int32_t tokens);

// sm_89 FP8 prefill routes; null off sm_89, below the family threshold, or when the
// EngineOptions/CLI gate is off.
[[nodiscard]] A8PrefillLaunch select_q4_a8_prefill_n4096_k5120(std::int32_t tokens);
[[nodiscard]] A8PrefillLaunch select_q4_a8_prefill_n5120_k6144(std::int32_t tokens);
[[nodiscard]] A8PrefillLaunch select_q4_a8_prefill_n6144_k5120(std::int32_t tokens);
[[nodiscard]] A8PrefillLaunch select_q4_a8_prefill_n7168_k5120(std::int32_t tokens);
[[nodiscard]] A8PrefillLaunch select_q4_a8_prefill_n34816_k5120(std::int32_t tokens);

} // namespace ninfer::ops::detail
