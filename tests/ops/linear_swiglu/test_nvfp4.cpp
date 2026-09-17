#include "core/weight.h"
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 4> kA16Cases{1, 4, 8, 16};
#if !defined(NINFER_SM89)
        // 255, 256 and 257 straddle the fused route floor. This change does not move that
        // route, but 257 reaches it through the baseline composition, whose own linear now
        // runs the ragged TMA path.
        constexpr std::array<std::int32_t, 17> kA4Cases{2,   4,   5,   16,  56,  64,  65,  96,  97,
                                                        112, 128, 129, 255, 256, 257, 512, 1024};
#endif
        int failures = 0;
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
#if !defined(NINFER_SM89)
        // NVFP4 W4A4 needs the Blackwell FP4 MMA; sm_89 builds compile the stub instead,
        // which rejects the route at runtime (see src/ops/nvfp4_w4a4_stubs.cpp).
        failures += run_profile("LinearSwiGLU NVFP4_A4",
                                {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4},
                                kA4Cases, std::array<std::int32_t, 4>{65, 97, 128, 129});
#endif
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
