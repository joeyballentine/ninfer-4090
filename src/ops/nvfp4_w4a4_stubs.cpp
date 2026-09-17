// NVFP4 W4A4 (TMA) kernel entry points for non-sm_120a builds.
//
// W4A4 contracts through the Blackwell block-scaled FP4 MMA (kind::mxf4nvf4) and, above the TMA
// floor, through TMA; the kernels that own both are excluded from non-120 builds (see
// src/ops/CMakeLists.txt). These stubs keep the dispatch layer linkable and turn any NVFP4 W4A4
// use on this architecture into a clear runtime error.
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void nvfp4_w4a4_unsupported() {
    throw std::runtime_error(
        "NVFP4 W4A4 execution requires the SM120 (Blackwell) TMA kernels, which are not "
        "compiled for this architecture");
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace,
                                Nvfp4ScaleLayout, cudaStream_t) {
    nvfp4_w4a4_unsupported();
}

// Every shape's A4 route resolves here on this architecture; see ops/linear/nvfp4/nvfp4_launch.cuh.
void nvfp4_a4_unsupported(const Weight&, Tensor&, Nvfp4W4a4Workspace, std::int32_t, cudaStream_t) {
    nvfp4_w4a4_unsupported();
}

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                  Tensor&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    nvfp4_w4a4_unsupported();
}

void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) {
    nvfp4_w4a4_unsupported();
}

void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    nvfp4_w4a4_unsupported();
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&,
                                     WorkspaceArena&, cudaStream_t) {
    nvfp4_w4a4_unsupported();
}

} // namespace ninfer::ops::detail
