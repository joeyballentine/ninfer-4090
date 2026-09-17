#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_geometry.h"

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8KSplitScaleAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class Q8KSplitActivationStage : std::uint8_t {
    // Stage the compile-time column extent; tiled calls zero-fill its inactive columns.
    ActiveOnly,
    // Stage the full MMA tile, including padding beyond the compile-time column extent.
    PaddedZero,
    // Stage only live columns. Inactive MMA columns are discarded by the store epilogue.
    RuntimeActive,
};

// Static shared bytes of Q8KSplitSharedStorage for a given warp count and token tile. The
// staging arm dominates every registered tile, but both are stated so the bound is the union's.
inline constexpr int q8_ksplit_shared_bytes(int warps, int tile_tokens, bool shared_scales) {
    const int group_k = warps * 64;
    const int staging = 16 * group_k                      // codes
                        + warps * tile_tokens * 64 * 2    // activations
                        + 16 * (shared_scales ? group_k / 16 : 1);
    const int partial = warps * (tile_tokens / 8) * 32 * 4 * 4;
    return staging > partial ? staging : partial;
}

// sm_89 (Ada) gives a block 48 KiB of static shared memory, against the 99 KiB the sm_120a
// schedules are tuned against. The staging layout is linear in the K-split warp count, so that
// is the single knob capped here: every shape keeps its measured token tile, cache policy and
// staging mode, and the contraction is unchanged - fewer warps own a wider K slice each and the
// CTA reduces fewer FP32 partials.
inline constexpr int kQ8KSplitStaticSharedLimit = 48 * 1024;

inline constexpr int q8_ksplit_fitted_warps(int warps, int tile_tokens, bool shared_scales) {
    while (warps > 4 &&
           q8_ksplit_shared_bytes(warps, tile_tokens, shared_scales) > kQ8KSplitStaticSharedLimit) {
        warps /= 2;
    }
    return warps;
}

template <int KWarps, int TileTokens, int MinBlocksPerSm, Q8KSplitScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8KSplitActivationStage ActivationStage = Q8KSplitActivationStage::ActiveOnly>
struct Q8KSplitSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48 || TileTokens == 56 || TileTokens == 64 ||
                  TileTokens == 72 || TileTokens == 80 || TileTokens == 88);
    static_assert(MinBlocksPerSm > 0);

#if defined(NINFER_SM89)
    static constexpr int kKWarps =
        q8_ksplit_fitted_warps(KWarps, TileTokens, ScaleAccess == Q8KSplitScaleAccess::Shared);
#else
    static constexpr int kKWarps            = KWarps;
#endif
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    static constexpr int kScaleBytesPerRow  = kGroupK / 16;
};

template <int TileTokens, int ActiveTokens>
using Q8KSplitDefaultSchedule = Q8KSplitSchedule<
    8, TileTokens, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
    (ActiveTokens > 4 ? Q8KSplitScaleAccess::Shared : Q8KSplitScaleAccess::Direct)>;

} // namespace ninfer::ops::detail
