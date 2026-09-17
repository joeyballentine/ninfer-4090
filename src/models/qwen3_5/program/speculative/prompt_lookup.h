#pragma once

#include "models/qwen3_5/program/round_buffers.h"

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5 {

// Longest and shortest n-gram orders the matcher considers. A two-token context repeats far too
// often to predict a continuation, so three is the shortest useful order.
inline constexpr std::uint32_t kPromptLookupMaximumOrder = 5;
inline constexpr std::uint32_t kPromptLookupMinimumOrder = 3;

struct PromptLookupDraft {
    std::uint32_t count = 0;
    std::array<TokenId, kMtpDecodeMaximumDrafts> tokens{};
};

// Draft the continuation of the most recent repetition of the sequence tail within the sequence's
// own committed token history.
//
// The ledger tail of length n is matched against every earlier n-gram ending at least n tokens
// before the end, longest order first: a length-5 match anywhere beats a length-4 match, and
// within one order the most recent occurrence wins. The tokens following the matched occurrence
// become the draft, truncated to draft_window. Without a match of at least the minimum order the
// result is empty.
//
// One backward scan computes this. The run of tokens matching the tail at a candidate position is
// that position's order, and because the orders are nested, the first position reaching the
// maximum order is also what a longest-first hierarchical search would return.
[[nodiscard]] inline PromptLookupDraft
find_prompt_lookup_draft(std::span<const TokenId> ledger, std::uint32_t draft_window) noexcept {
    PromptLookupDraft result{};
    const std::size_t total = ledger.size();
    const std::size_t drafts =
        draft_window < kMtpDecodeMaximumDrafts ? draft_window : kMtpDecodeMaximumDrafts;
    if (drafts == 0 || total < 2 * kPromptLookupMinimumOrder) { return result; }

    const auto emit = [&](std::size_t position) noexcept {
        const std::size_t available = total - 1 - position;
        const std::size_t count     = available < drafts ? available : drafts;
        result.count                = static_cast<std::uint32_t>(count);
        for (std::size_t i = 0; i < count; ++i) { result.tokens[i] = ledger[position + 1 + i]; }
    };

    std::size_t best_position = 0;
    std::size_t best_order    = 0;
    // A shorter order needs less room between its occurrence and the tail, so the scan starts at
    // the last position that can still carry a minimum-order continuation.
    for (std::size_t scan = total - kPromptLookupMinimumOrder; scan > 0; --scan) {
        const std::size_t position = scan - 1;
        std::size_t run            = 0;
        while (run < kPromptLookupMaximumOrder && run <= position &&
               ledger[position - run] == ledger[total - 1 - run]) {
            ++run;
        }
        const std::size_t room  = total - 1 - position;
        const std::size_t order = run < room ? run : room;
        if (order < kPromptLookupMinimumOrder) { continue; }
        if (order == kPromptLookupMaximumOrder) {
            emit(position);
            return result;
        }
        if (order > best_order) {
            best_order    = order;
            best_position = position;
        }
    }
    if (best_order != 0) { emit(best_position); }
    return result;
}

} // namespace ninfer::models::qwen3_5
