#pragma once

// Admission and caller-owned scratch for the Ada (sm_89) FP8 prefill routes of the groupwise
// row-split Linear formats. See docs/maintainer/ada-fp8-prefill.md.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The A8 prefill routes reuse the row-scaled FP8 family's activation materialization unchanged:
// the same [T,K] E4M3 code plane, the same absmax/448 FP32 row scale, and the same two-plane
// workspace. Only the weight operand differs, so there is nothing to fork here.
using A8PrefillWorkspace = Fp8A8Workspace;

// Every A8 prefill route is selected as a shape-owned function pointer, like its A16 peers, and
// additionally receives the call-scoped arena that holds the materialized activation.
using A8PrefillLaunch = void (*)(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                 cudaStream_t);

// Below this extent the contraction is not tensor-core bound at these shapes, and the extra
// activation materialization pass is not repaid. 128 is also one token tile of the route.
inline constexpr std::int32_t kA8PrefillMinTokens = 128;

// One activation-quantizer instance is compiled per registered K.
[[nodiscard]] constexpr bool a8_prefill_supported_k(std::int32_t k) noexcept {
    return k == 5120 || k == 6144 || k == 17408;
}

// Route admission. Off sm_89 the family is not compiled in at all; on sm_89 it still requires the
// A8 activation permission and the EngineOptions/CLI opt-in, which defaults to off.
[[nodiscard]] inline bool a8_prefill_admits(LinearPolicy policy, std::int32_t k,
                                            std::int32_t tokens) noexcept {
#if defined(NINFER_SM89)
    return allows_a8(policy) && ninfer::ops::prefill_a8_routes_enabled() &&
           a8_prefill_supported_k(k) && tokens >= kA8PrefillMinTokens;
#else
    (void)policy;
    (void)k;
    (void)tokens;
    return false;
#endif
}

[[nodiscard]] inline std::size_t a8_prefill_workspace_capacity_bytes(std::int32_t tokens,
                                                                     std::int32_t k) {
    return fp8_a8_workspace_capacity_bytes(tokens, k);
}

} // namespace ninfer::ops::detail
