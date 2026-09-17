#include "serve/serve_metrics.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdio>
#include <system_error>
#include <utility>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

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
                                 "Terminal requests the server failed or rejected."};
constexpr Family kRequestsCancelled{"ninfer:requests_cancelled_total", "counter",
                                    "Terminal requests whose client disconnected or cancelled."};
constexpr Family kReasoningTokens{"ninfer:reasoning_tokens_total", "counter",
                                  "Completion tokens attributed to reasoning content."};
constexpr Family kPrefixCacheHits{"ninfer:prefix_cache_hit_tokens_total", "counter",
                                  "Prompt tokens served from a reused KV prefix instead of "
                                  "being recomputed."};
constexpr Family kDraftTokens{"ninfer:draft_tokens_total", "counter",
                              "Speculative draft tokens proposed."};
constexpr Family kDraftAccepted{"ninfer:draft_accepted_tokens_total", "counter",
                                "Speculative draft tokens accepted by verification."};
constexpr Family kProcessing{"llamacpp:requests_processing", "gauge",
                             "Accepted requests currently occupying an execution lane."};
constexpr Family kDeferred{"llamacpp:requests_deferred", "gauge",
                           "Accepted requests waiting in the ingress FIFO for a lane."};
constexpr Family kPrefilling{"ninfer:requests_prefilling", "gauge",
                             "Lane-resident requests whose prompt is still being evaluated."};
constexpr Family kDecodeReady{"ninfer:requests_decode_ready", "gauge",
                              "Lane-resident requests eligible for the next decode round."};
constexpr Family kMaterializing{"ninfer:requests_materializing", "gauge",
                                "Requests whose model state is being materialized onto a lane."};

// Persistent prompt-cache tier. `lookups`/`hits` count disk index probes made after Device and
// Host missed, so `hits - restores` is the share of index hits that could not be brought back.
constexpr Family kPromptCacheLookups{"ninfer:prompt_cache_lookups_total", "counter",
                                     "Disk index probes made after the Device and Host context "
                                     "tiers missed."};
constexpr Family kPromptCacheHits{"ninfer:prompt_cache_hits_total", "counter",
                                  "Disk index probes that matched a published record."};
constexpr Family kPromptCacheRestores{"ninfer:prompt_cache_restores_total", "counter",
                                      "Disk records read back and adopted as a resident context "
                                      "source."};
constexpr Family kPromptCacheRestoreFailures{
    "ninfer:prompt_cache_restore_failures_total", "counter",
    "Matched records the Program declined or could not read back; the request falls back to "
    "prefill."};
constexpr Family kPromptCacheRestoredBytes{"ninfer:prompt_cache_restored_bytes_total", "counter",
                                           "Payload bytes read back from the store."};
constexpr Family kPromptCacheSpillRequests{"ninfer:prompt_cache_spill_requests_total", "counter",
                                           "Checkpoints offered to the disk tier by write-behind "
                                           "publication or eviction."};
constexpr Family kPromptCacheSpills{"ninfer:prompt_cache_spills_total", "counter",
                                    "Offered checkpoints written and published as a record."};
constexpr Family kPromptCacheSpillsDropped{
    "ninfer:prompt_cache_spills_dropped_total", "counter",
    "Offered checkpoints dropped by queue pressure, request-arrival cancellation, a vanished "
    "owner, or an I/O failure."};
constexpr Family kPromptCacheSpilledBytes{"ninfer:prompt_cache_spilled_bytes_total", "counter",
                                          "Payload bytes staged for published records."};
constexpr Family kPromptCacheRecords{"ninfer:prompt_cache_records", "gauge",
                                     "Records currently published in the store."};
constexpr Family kPromptCacheEvictions{"ninfer:prompt_cache_evictions_total", "counter",
                                       "Records retired by the size cap, least recently used "
                                       "first."};
constexpr Family kPromptCacheCompactions{"ninfer:prompt_cache_compactions_total", "counter",
                                         "Mark-and-sweep rewrites of the extent file."};
constexpr Family kPromptCacheLiveBytes{"ninfer:prompt_cache_live_bytes", "gauge",
                                       "Extent bytes referenced by a published record; this is "
                                       "what --prompt-cache-max-bytes caps."};
constexpr Family kPromptCacheFileBytes{"ninfer:prompt_cache_file_bytes", "gauge",
                                       "Bytes the store occupies on disk, including extents not "
                                       "yet reclaimed by compaction."};

} // namespace

ExecutorGauges make_executor_gauges(const ServeOptions& options,
                                    const ninfer::RuntimeStats& stats) {
    return ExecutorGauges{
        .max_concurrency = options.max_concurrency,
        .max_context     = options.max_context,
        .running         = stats.running_requests,
        .prefilling      = stats.prefilling_requests,
        .decode_ready    = stats.decode_ready_requests,
        .waiting         = stats.waiting_requests,
        .materializing   = stats.materializing_requests,
        .speculative     = options.speculative.backend != ninfer::SpeculativeBackend::None,
        .prompt_cache =
            PromptCacheGauges{
                .lookups          = stats.prompt_cache_lookups,
                .hits             = stats.prompt_cache_hits,
                .restores         = stats.prompt_cache_restores,
                .restore_failures = stats.prompt_cache_restore_failures,
                .restored_bytes   = stats.prompt_cache_restored_bytes,
                .spill_requests   = stats.prompt_cache_spill_requests,
                .spills           = stats.prompt_cache_spills,
                .spills_dropped   = stats.prompt_cache_spills_dropped,
                .spilled_bytes    = stats.prompt_cache_spilled_bytes,
                .records          = stats.prompt_cache_records,
                .evictions        = stats.prompt_cache_evictions,
                .compactions      = stats.prompt_cache_compactions,
                .live_bytes       = stats.prompt_cache_live_bytes,
                .file_bytes       = stats.prompt_cache_file_bytes,
            },
    };
}

std::string render_slots_json(const ExecutorGauges& gauges,
                              const std::vector<InFlightRequest>& in_flight) {
    // The executor owns how many lanes are occupied; it does not publish which request sits on
    // which lane. Admission is bounded FIFO without preemption, so the `running` oldest accepted
    // requests are exactly the lane-resident ones, and the remainder is the pending tail.
    const std::size_t resident =
        std::min<std::size_t>(in_flight.size(), std::min(gauges.running, gauges.max_concurrency));

    Json slots = Json::array();
    for (std::uint32_t lane = 0; lane < gauges.max_concurrency; ++lane) {
        const bool occupied = lane < resident;
        Json slot{{"id", lane},
                  {"state", occupied ? "processing" : "idle"},
                  {"n_ctx", gauges.max_context},
                  {"speculative", gauges.speculative}};
        if (occupied) {
            const InFlightRequest& request = in_flight[lane];
            slot["request_id"]             = request.request_id;
            slot["protocol"]               = request.protocol;
            slot["model"]                  = request.model;
            slot["n_prompt_tokens"]        = request.prompt_tokens;
            slot["elapsed_seconds"]        = request.elapsed_seconds;
        } else {
            slot["request_id"]      = nullptr;
            slot["n_prompt_tokens"] = 0;
        }
        slots.push_back(std::move(slot));
    }

    Json pending = Json::array();
    for (std::size_t index = resident; index < in_flight.size(); ++index) {
        const InFlightRequest& request = in_flight[index];
        pending.push_back(Json{{"request_id", request.request_id},
                               {"protocol", request.protocol},
                               {"model", request.model},
                               {"n_prompt_tokens", request.prompt_tokens},
                               {"elapsed_seconds", request.elapsed_seconds}});
    }

    return Json{{"slots", std::move(slots)},
                {"pending", std::move(pending)},
                {"requests_processing", resident},
                {"requests_deferred", in_flight.size() - resident}}
        .dump();
}

void ServeMetrics::begin_request(const RequestLogContext& context) {
    const std::lock_guard lock(mutex_);
    in_flight_[context.id] = InFlightEntry{
        .protocol      = context.protocol,
        .model         = context.model,
        .prompt_tokens = context.prompt_tokens,
        .started       = std::chrono::steady_clock::now(),
    };
}

void ServeMetrics::end_request(std::uint64_t request_id) {
    const std::lock_guard lock(mutex_);
    in_flight_.erase(request_id);
}

std::vector<InFlightRequest> ServeMetrics::in_flight_snapshot() const {
    const auto now = std::chrono::steady_clock::now();
    const std::lock_guard lock(mutex_);
    std::vector<InFlightRequest> snapshot;
    snapshot.reserve(in_flight_.size());
    for (const auto& [request_id, entry] : in_flight_) {
        snapshot.push_back(InFlightRequest{
            .request_id      = request_id,
            .protocol        = entry.protocol,
            .model           = entry.model,
            .prompt_tokens   = entry.prompt_tokens,
            .elapsed_seconds = std::chrono::duration<double>(now - entry.started).count(),
        });
    }
    return snapshot;
}

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

void ServeMetrics::record_failure(const RequestFailure& failure) {
    // An agent client that abandons a request is a normal outcome, not a server fault; keeping it
    // out of the failure counter is what makes that counter usable as an alerting signal.
    const bool cancelled = failure.classification == RequestFailureClass::ClientDisconnected;
    const std::lock_guard lock(mutex_);
    ++requests_total_;
    if (cancelled) {
        ++requests_cancelled_total_;
    } else {
        ++requests_failed_total_;
    }
}

void ServeMetrics::record_rejected() {
    const std::lock_guard lock(mutex_);
    ++requests_total_;
    ++requests_failed_total_;
}

std::string ServeMetrics::render(const ExecutorGauges& gauges) const {
    const std::lock_guard lock(mutex_);
    std::string out;
    out.reserve(2048);
    append_sample(out, kPromptTokens, prompt_tokens_total_);
    append_sample(out, kPromptSeconds, prompt_seconds_total_);
    append_sample(out, kPredictedTokens, tokens_predicted_total_);
    append_sample(out, kPredictedSeconds, tokens_predicted_seconds_total_);
    append_sample(out, kRequests, requests_total_);
    append_sample(out, kRequestsFailed, requests_failed_total_);
    append_sample(out, kRequestsCancelled, requests_cancelled_total_);
    append_sample(out, kReasoningTokens, reasoning_tokens_total_);
    append_sample(out, kPrefixCacheHits, prefix_cache_hit_tokens_total_);
    append_sample(out, kDraftTokens, speculative_draft_tokens_total_);
    append_sample(out, kDraftAccepted, speculative_accepted_tokens_total_);
    append_sample(out, kProcessing, static_cast<std::uint64_t>(gauges.running));
    append_sample(out, kDeferred, static_cast<std::uint64_t>(gauges.waiting));
    append_sample(out, kPrefilling, static_cast<std::uint64_t>(gauges.prefilling));
    append_sample(out, kDecodeReady, static_cast<std::uint64_t>(gauges.decode_ready));
    append_sample(out, kMaterializing, static_cast<std::uint64_t>(gauges.materializing));
    const PromptCacheGauges& cache = gauges.prompt_cache;
    append_sample(out, kPromptCacheLookups, cache.lookups);
    append_sample(out, kPromptCacheHits, cache.hits);
    append_sample(out, kPromptCacheRestores, cache.restores);
    append_sample(out, kPromptCacheRestoreFailures, cache.restore_failures);
    append_sample(out, kPromptCacheRestoredBytes, cache.restored_bytes);
    append_sample(out, kPromptCacheSpillRequests, cache.spill_requests);
    append_sample(out, kPromptCacheSpills, cache.spills);
    append_sample(out, kPromptCacheSpillsDropped, cache.spills_dropped);
    append_sample(out, kPromptCacheSpilledBytes, cache.spilled_bytes);
    append_sample(out, kPromptCacheRecords, cache.records);
    append_sample(out, kPromptCacheEvictions, cache.evictions);
    append_sample(out, kPromptCacheCompactions, cache.compactions);
    append_sample(out, kPromptCacheLiveBytes, cache.live_bytes);
    append_sample(out, kPromptCacheFileBytes, cache.file_bytes);
    return out;
}

} // namespace ninfer::serve
