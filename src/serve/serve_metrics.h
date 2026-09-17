#pragma once

// Cumulative serving counters behind `GET /metrics`, rendered in the Prometheus text exposition
// format. This layer owns no accounting of its own: every counter is fed from the same terminal
// request funnel that drives operational_log.* and the JSONL measurement log, so a request is
// counted exactly once, in every protocol and in both consumer modes.
//
// The `llamacpp:`-prefixed families reproduce llama.cpp's `--metrics` semantics - computed prefill
// tokens (prefix-cache hits excluded) billed against prefill wall time, committed decode tokens
// against decode wall time - so an existing llama.cpp scrape config reads this server unchanged.
// The `ninfer:` families report what llama.cpp has no equivalent for: prefix-cache reuse,
// speculative draft acceptance, and terminal request classification.

#include "serve/generation_service.h"
#include "serve/request_events.h"

#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

class ServeMetrics {
public:
    // One completed request, taken from the request-done funnel.
    void record_done(const GenerationOutcome& outcome);

    // A request that parsed and then failed, and a request rejected during validation. Both are
    // terminal, so `ninfer:requests_total` counts every request exactly once.
    void record_failure(const RequestFailure& failure);
    void record_rejected();

    // One complete Prometheus text body, without HTTP framing. Ends with a newline.
    [[nodiscard]] std::string render() const;

private:
    mutable std::mutex mutex_;
    std::uint64_t requests_total_                    = 0;
    std::uint64_t requests_failed_total_             = 0;
    std::uint64_t prompt_tokens_total_               = 0;
    double prompt_seconds_total_                     = 0.0;
    std::uint64_t tokens_predicted_total_            = 0;
    double tokens_predicted_seconds_total_           = 0.0;
    std::uint64_t reasoning_tokens_total_            = 0;
    std::uint64_t prefix_cache_hit_tokens_total_     = 0;
    std::uint64_t speculative_draft_tokens_total_    = 0;
    std::uint64_t speculative_accepted_tokens_total_ = 0;
};

} // namespace ninfer::serve
