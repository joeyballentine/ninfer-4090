// ninfer::ops::detail - sm_89 rk4v4-e8 small-T instantiations (rk4v4 with the E8 lattice K encode).
// Split out of small_t.cu so each sm_89 INT8 KV storage owns one ptxas job.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch_impl.cuh"

#if defined(NINFER_SM89)

namespace ninfer::ops::detail {

NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE(CausalSmallTI8Storage::RK4V4E8)

} // namespace ninfer::ops::detail

#endif // NINFER_SM89
