#include "serve/serve_metrics.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <system_error>

namespace ninfer::serve {
namespace {

struct Family {
    const char* name;
    const char* type;
    const char* help;
};

void append_header(std::string& out, const Family& family) {
    out += "# HELP ";
    out += family.name;
    out += ' ';
    out += family.help;
    out += "\n# TYPE ";
    out += family.name;
    out += ' ';
    out += family.type;
    out += '\n';
}

void append_sample(std::string& out, const Family& family, std::uint64_t value) {
    append_header(out, family);
    std::array<char, 24> digits{};
    const auto converted = std::to_chars(digits.data(), digits.data() + digits.size(), value);
    out += family.name;
    out += ' ';
    out.append(digits.data(), static_cast<std::size_t>(converted.ptr - digits.data()));
    out += '\n';
}

void append_sample(std::string& out, const Family& family, double value) {
    append_header(out, family);
    std::array<char, 40> text{};
    // Prometheus reads a plain decimal; six fractional digits resolve sub-millisecond phases
    // without exporting float noise from the wall clocks these seconds come from.
    const int written = std::snprintf(text.data(), text.size(), "%.6f", value);
    out += family.name;
    out += ' ';
    if (written > 0) { out.append(text.data(), static_cast<std::size_t>(written)); }
    out += '\n';
}

constexpr Family kPromptTokens{"llamacpp:prompt_tokens_total", "counter",
                               "Prompt tokens actually evaluated by prefill, excluding tokens "
                               "served from a reused prefix."};
constexpr Family kPromptSeconds{"llamacpp:prompt_seconds_total", "counter",
                                "Wall seconds spent in prefill across completed requests."};
constexpr Family kPredictedTokens{"llamacpp:tokens_predicted_total", "counter",
                                  "Completion tokens committed by decode."};
constexpr Family kPredictedSeconds{"llamacpp:tokens_predicted_seconds_total", "counter",
                                   "Wall seconds spent in decode across completed requests."};
constexpr Family kRequests{"ninfer:requests_total", "counter",
                           "Terminal requests, whether they completed, failed, or were rejected."};
constexpr Family kRequestsFailed{"ninfer:requests_failed_total", "counter",
                                 "Terminal requests that did not produce a completion."};
constexpr Family kReasoningTokens{"ninfer:reasoning_tokens_total", "counter",
                                  "Completion tokens attributed to reasoning content."};
constexpr Family kPrefixCacheHits{"ninfer:prefix_cache_hit_tokens_total", "counter",
                                  "Prompt tokens served from a reused KV prefix instead of "
                                  "being recomputed."};
constexpr Family kDraftTokens{"ninfer:draft_tokens_total", "counter",
                              "Speculative draft tokens proposed."};
constexpr Family kDraftAccepted{"ninfer:draft_accepted_tokens_total", "counter",
                                "Speculative draft tokens accepted by verification."};

} // namespace

void ServeMetrics::record_done(const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const std::uint64_t cached       = metrics.prefix_cache_hit_tokens;
    const std::uint64_t prompt =
        outcome.prompt_tokens > 0 ? static_cast<std::uint64_t>(outcome.prompt_tokens) : 0;
    // A reused prefix is reported against the whole prompt; clamp rather than wrap when the two
    // accountings disagree at a boundary.
    const std::uint64_t computed_prefill = prompt > cached ? prompt - cached : 0;

    const std::lock_guard lock(mutex_);
    ++requests_total_;
    prompt_tokens_total_ += computed_prefill;
    prompt_seconds_total_ += std::max(0.0, metrics.prefill_seconds);
    tokens_predicted_total_ +=
        outcome.completion_tokens > 0 ? static_cast<std::uint64_t>(outcome.completion_tokens) : 0;
    tokens_predicted_seconds_total_ += std::max(0.0, metrics.decode_seconds);
    reasoning_tokens_total_ +=
        outcome.reasoning_tokens > 0 ? static_cast<std::uint64_t>(outcome.reasoning_tokens) : 0;
    prefix_cache_hit_tokens_total_ += std::min(prompt, cached);
    speculative_draft_tokens_total_ += metrics.speculative_draft_tokens;
    speculative_accepted_tokens_total_ += metrics.speculative_accepted_tokens;
}

void ServeMetrics::record_failure(const RequestFailure&) {
    const std::lock_guard lock(mutex_);
    ++requests_total_;
    ++requests_failed_total_;
}

void ServeMetrics::record_rejected() {
    const std::lock_guard lock(mutex_);
    ++requests_total_;
    ++requests_failed_total_;
}

std::string ServeMetrics::render() const {
    const std::lock_guard lock(mutex_);
    std::string out;
    out.reserve(2048);
    append_sample(out, kPromptTokens, prompt_tokens_total_);
    append_sample(out, kPromptSeconds, prompt_seconds_total_);
    append_sample(out, kPredictedTokens, tokens_predicted_total_);
    append_sample(out, kPredictedSeconds, tokens_predicted_seconds_total_);
    append_sample(out, kRequests, requests_total_);
    append_sample(out, kRequestsFailed, requests_failed_total_);
    append_sample(out, kReasoningTokens, reasoning_tokens_total_);
    append_sample(out, kPrefixCacheHits, prefix_cache_hit_tokens_total_);
    append_sample(out, kDraftTokens, speculative_draft_tokens_total_);
    append_sample(out, kDraftAccepted, speculative_accepted_tokens_total_);
    return out;
}

} // namespace ninfer::serve
