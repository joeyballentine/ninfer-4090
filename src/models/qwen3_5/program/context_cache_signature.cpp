#include "models/qwen3_5/program/context_cache_signature.h"

#include <array>
#include <cstdio>

namespace ninfer::models::qwen3_5 {
namespace {

[[nodiscard]] const char* speculative_backend_token(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "unknown";
}

// The schedule spec spells a two-tier schedule `head:N,tail`; neither separator survives a
// directory name, so both become `-` here instead of being rewritten later by the store.
[[nodiscard]] std::string kv_schedule_token(const KvCacheSchedule& schedule) {
    std::string spec = kv_cache_schedule_spec(schedule);
    for (char& character : spec) {
        if (character == ':' || character == ',') { character = '-'; }
    }
    return spec;
}

} // namespace

std::string context_cache_identity_digest(std::string_view text) noexcept {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (const char character : text) {
        hash ^= static_cast<std::uint8_t>(character);
        hash *= 0x100000001b3ULL;
    }
    std::array<char, 17> digits{};
    (void)std::snprintf(digits.data(), digits.size(), "%016llx",
                        static_cast<unsigned long long>(hash));
    return std::string(digits.data(), 16);
}

std::string context_cache_signature(const ContextCacheSignatureFacts& facts) {
    std::string out   = "q35-ctx1";
    const auto append = [&out](std::string_view key, std::string_view value) {
        out += '.';
        out += key;
        out += value;
    };
    const auto append_number = [&](std::string_view key, std::uint64_t value) {
        append(key, std::to_string(value));
    };
    append("a", context_cache_identity_digest(facts.artifact_identity));
    append("kv", kv_schedule_token(facts.kv_storage));
    append_number("t", facts.kv_storage.identity_tag());
    append("sb", speculative_backend_token(facts.speculative_backend));
    append("ph", facts.proposal_head == ProposalHead::Full ? "full" : "opt");
    append_number("c", facts.max_context);
    append_number("k", facts.kv_capacity);
    append_number("mp", facts.main_page_tokens);
    append_number("mn", facts.main_plane_count);
    append_number("mb", facts.main_page_bytes);
    append_number("bp", facts.backend_page_tokens);
    append_number("bn", facts.backend_plane_count);
    append_number("bb", facts.backend_page_bytes);
    append_number("s", facts.state_image_bytes);
    append_number("v", facts.vision_enabled);
    return out;
}

} // namespace ninfer::models::qwen3_5
