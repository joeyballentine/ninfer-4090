#include "core/weight.h"
#include "ops/linear/linear_test_common.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <utility>
#include <iostream>

// Qualification for the sm_89 FP8 prefill route of Q4_G64_FP16 Linear.
//
// Tolerance. The suite-owned A8 criterion (ReductionCriterion{0.04, bf16 u, 0.06} in
// linear_test_common.cpp) is the right one for this route, and the derivation is the same one it
// was written for:
//
//   * Weight side: exact. A Q4 code is a 4-bit two's-complement integer in [-8,7]; E4M3 carries
//     three mantissa bits, so every integer up to 16 is an E4M3 value and the decode is a recode,
//     not a quantization. The binary16 group scale is applied in FP32 after the MMA, so the only
//     weight-side error is one FP32 multiply (2^-24 relative), two orders below BF16 output
//     rounding. The route therefore adds nothing to the weight error of the A16 route.
//   * Activation side: x is quantized per token row as round_e4m3(x / s) with s = absmax(row)/448.
//     E4M3 normals have a 2^-3 relative ulp, so the half-ulp bound is |dx| <= 2^-4 |x| = 0.0625|x|;
//     the subnormal floor contributes at most s * 2^-10 ~ absmax * 2^-19 and is negligible.
//     Treating the per-element relative errors as independent and uniform, their standard
//     deviation is 2^-4/sqrt(3) = 0.036.
//   * Through the dot product both the signal and the error accumulate as sqrt(K), so the relative
//     L2 error of an output column stays at ~0.036 and does not grow with K. 0.04 is that bound
//     with a small margin; the 1.5x gross pointwise cap absorbs columns whose dot product
//     cancels.
//
// The route is not reachable off sm_89, below T = 128, or with the gate off, so this test enables
// the gate and skips when the device is not Ada.

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

bool ada_device() {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) { return false; }
    return properties.major == 8 && properties.minor == 9;
}

int check_workspace_contract(std::int32_t n, std::int32_t k) {
    int failures = 0;
    const auto capacity = [&](ops::LinearPolicy policy, std::int32_t t) {
        return ops::linear_workspace_capacity_bytes(QType::Q4_G64_FP16, n, k, policy, t, t);
    };
    // One E4M3 code plane plus one FP32 row scale, each aligned to 256 bytes.
    const std::size_t payload_2048 =
        static_cast<std::size_t>(2048) * static_cast<std::size_t>(k) + 2048U * sizeof(float);
    const std::size_t reported = capacity(ops::LinearPolicy::AllowA8, 2048);
    if (reported < payload_2048 || reported > payload_2048 + 512U ||
        capacity(ops::LinearPolicy::AllowA4, 2048) != reported ||
        capacity(ops::LinearPolicy::AllowA8, 128) == 0 ||
        capacity(ops::LinearPolicy::AllowA8, 127) != 0 ||
        capacity(ops::LinearPolicy::A16Only, 2048) != 0) {
        std::cerr << "Q4 A8 prefill workspace contract mismatch for N=" << n << " K=" << k << '\n';
        ++failures;
    }
    // An interval is sized by its top extent, and a purely sub-threshold interval needs nothing.
    if (ops::linear_workspace_capacity_bytes(QType::Q4_G64_FP16, n, k, ops::LinearPolicy::AllowA8,
                                             1, 2048) != reported ||
        ops::linear_workspace_capacity_bytes(QType::Q4_G64_FP16, n, k, ops::LinearPolicy::AllowA8,
                                             1, 127) != 0) {
        std::cerr << "Q4 A8 prefill workspace interval mismatch for N=" << n << " K=" << k << '\n';
        ++failures;
    }
    return failures;
}

int run_q4_a8_prefill() {
    // T = 128 is the route boundary: b-1 falls back to A16, b and b+1 take the new route, and
    // 512 / 2048 are the prefill anchors.
    constexpr std::array a8_invocations{
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{129, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{512, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{2048, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array a16_invocations{
        Invocation{127, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };

    int failures = 0;
    for (const auto shape : {std::pair<std::int32_t, std::int32_t>{34816, 5120},
                             std::pair<std::int32_t, std::int32_t>{6144, 5120}}) {
        const std::uint32_t seed = 911U + static_cast<std::uint32_t>(shape.first % 1000);
        failures += run_shape("Q4_A8_PREFILL", ActivationCompute::A8, make_q4_g64_fp16_weight,
                              {shape.first, shape.second, seed, Comparison::Sampled, true,
                               a8_invocations});
        // Below the boundary the gate changes nothing, so the A16 criterion still holds.
        failures += run_shape("Q4_A8_PREFILL_BELOW", ActivationCompute::A16,
                              make_q4_g64_fp16_weight,
                              {shape.first, shape.second, seed + 1U, Comparison::Sampled, true,
                               a16_invocations});
        failures += check_workspace_contract(shape.first, shape.second);
    }
    return failures;
}

} // namespace

int main() {
#if !defined(NINFER_SM89)
    std::cout << "SKIP: Q4 A8 prefill is an sm_89 build route\n";
    return 77;
#else
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (!ada_device()) {
        std::cout << "SKIP: Q4 A8 prefill requires an sm_89 device\n";
        return 77;
    }
    try {
        ninfer::ops::set_prefill_a8_routes_enabled(true);
        const int failures = run_q4_a8_prefill();
        ninfer::ops::set_prefill_a8_routes_enabled(false);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q4 A8 prefill Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q4 A8 prefill Linear: " << error.what() << '\n';
        return 1;
    }
#endif
}
