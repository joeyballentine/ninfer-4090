// ninfer::ops::detail - sm_89 rk8v4 small-T instantiations (int8 rotated K, packed int4 rotated V).
// Split out of small_t.cu so each sm_89 INT8 KV storage owns one ptxas job.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch_impl.cuh"

#if defined(NINFER_SM89)

namespace ninfer::ops::detail {

NINFER_CAUSAL_SMALL_T_I8_INSTANTIATE(CausalSmallTI8Storage::RotatedInt8KeyInt4ValueGroup64)

} // namespace ninfer::ops::detail

#endif // NINFER_SM89
