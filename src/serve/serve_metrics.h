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

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

// Engine-owned occupancy, read from the Engine's published RuntimeStats snapshot. The serve layer
// never recomputes these numbers: whether a submitted request currently occupies an execution lane
// is the executor's answer, not the HTTP layer's.
struct ExecutorGauges {
    std::uint32_t max_concurrency = 0;
    std::uint32_t max_context     = 0;
    std::uint32_t running         = 0;
    std::uint32_t prefilling      = 0;
    std::uint32_t decode_ready    = 0;
    std::uint32_t waiting         = 0;
    std::uint32_t materializing   = 0;
    bool speculative              = false;
};

[[nodiscard]] ExecutorGauges make_executor_gauges(const ServeOptions& options,
                                                  const ninfer::RuntimeStats& stats);

// One accepted request that has not reached a terminal boundary yet.
struct InFlightRequest {
    std::uint64_t request_id = 0;
    std::string protocol;
    std::string model;
    int prompt_tokens      = 0;
    double elapsed_seconds = 0.0;
};

// `GET /slots`: the executor's lane table, with the pending FIFO tail that has been accepted but
// does not occupy a lane yet.
[[nodiscard]] std::string render_slots_json(const ExecutorGauges& gauges,
                                            const std::vector<InFlightRequest>& in_flight);

class ServeMetrics {
public:
    // In-flight bookkeeping on the same request funnel as the counters. `end_request` is
    // idempotent and is reached from the done, failure, and rejection paths alike, so no terminal
    // outcome can leave a permanently occupied entry behind.
    void begin_request(const RequestLogContext& context);
    void end_request(std::uint64_t request_id);

    // Accepted requests that have not terminated, oldest first. Request ids are assigned in
    // arrival order, so this is the ingress FIFO order the executor admits from.
    [[nodiscard]] std::vector<InFlightRequest> in_flight_snapshot() const;

    // One completed request, taken from the request-done funnel.
    void record_done(const GenerationOutcome& outcome);

    // A request that parsed and then failed, and a request rejected during validation. Both are
    // terminal, so `ninfer:requests_total` counts every request exactly once.
    void record_failure(const RequestFailure& failure);
    void record_rejected();

    // One complete Prometheus text body, without HTTP framing. Ends with a newline. The executor
    // gauges are passed in rather than cached so a scrape reports the Engine's current occupancy
    // instead of whatever the last terminal request observed.
    [[nodiscard]] std::string render(const ExecutorGauges& gauges) const;

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

    struct InFlightEntry {
        std::string protocol;
        std::string model;
        int prompt_tokens = 0;
        std::chrono::steady_clock::time_point started;
    };

    std::map<std::uint64_t, InFlightEntry> in_flight_;
};

} // namespace ninfer::serve
