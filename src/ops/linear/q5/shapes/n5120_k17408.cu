#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_a8_prefill_launch.h"
#include "ops/linear/q5/q5_ksplit_launch.cuh"

namespace ninfer::ops::detail {

Q5Launch select_q5_n5120_k17408(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_simt_r8_c4;
    if (tokens <= 2) return launch_q5_ksplit<17408, 2, 2>;
    if (tokens <= 3) return launch_q5_ksplit<17408, 3, 2>;
    if (tokens <= 4) return launch_q5_ksplit<17408, 4, 2>;
    if (tokens <= 5) return launch_q5_ksplit<17408, 5, 2>;
    if (tokens <= 6) return launch_q5_ksplit<17408, 6, 2>;
    if (tokens <= 24) return launch_q5_simt_r8_c8;
    return launch_q5_mma_r64_c128;
}


// The FP8 prefill route claims the whole admitted interval at this shape: one 64-row x 128-token
// tile, which is the tile the BF16 prefill route already settles on here. Below the family
// threshold selection falls back to the A16 routes above.
A8PrefillLaunch select_q5_a8_prefill_n5120_k17408(std::int32_t tokens) {
#if defined(NINFER_SM89)
    if (tokens >= kA8PrefillMinTokens) { return launch_q5_a8_prefill_r64_c128; }
#else
    (void)tokens;
#endif
    return nullptr;
}

} // namespace ninfer::ops::detail
