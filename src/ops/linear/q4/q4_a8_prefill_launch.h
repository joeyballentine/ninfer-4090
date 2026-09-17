#pragma once

#include "ops/linear/a8_prefill/a8_prefill_plan.h"

namespace ninfer::ops::detail {

void launch_q4_a8_prefill_r64_c128(const Tensor& x, const Weight& w, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
