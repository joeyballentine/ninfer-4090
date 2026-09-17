#include "ops/linear/q5/q5_a8_prefill_launch.h"

#include "ops/linear/a8_prefill/a8_prefill_launch.cuh"
#include "ops/linear/q5/q5_a8_prefill_codec.cuh"

namespace ninfer::ops::detail {
namespace {
using Schedule = A8PrefillR64C128<Q5A8PrefillCodec::kHighBytesPerGroup>;
} // namespace

void launch_q5_a8_prefill_r64_c128(const Tensor& x, const Weight& w, Tensor& out,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    launch_a8_prefill_linear<Q5A8PrefillCodec, Schedule>(x, w, out, workspace, stream);
}

} // namespace ninfer::ops::detail
