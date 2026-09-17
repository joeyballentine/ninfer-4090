#include "serve/serve_metrics.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using ninfer::serve::ExecutorGauges;
using ninfer::serve::GenerationOutcome;
using ninfer::serve::InFlightRequest;
using ninfer::serve::RequestLogContext;
using ninfer::serve::RequestFailure;
using ninfer::serve::RequestFailureClass;
using ninfer::serve::RequestFailurePhase;
using ninfer::serve::render_slots_json;
using ninfer::serve::ServeMetrics;

ExecutorGauges gauges(std::uint32_t max_concurrency, std::uint32_t running, std::uint32_t waiting) {
    return ExecutorGauges{.max_concurrency = max_concurrency,
                          .max_context     = 65536,
                          .running         = running,
                          .prefilling      = running != 0 ? 1U : 0U,
                          .decode_ready    = 0,
                          .waiting         = waiting,
                          .materializing   = 0,
                          .speculative     = true};
}

RequestLogContext started(std::uint64_t id, const char* protocol, int prompt_tokens) {
    RequestLogContext context;
    context.id            = id;
    context.protocol      = protocol;
    context.model         = "qwen3.5";
    context.prompt_tokens = prompt_tokens;
    return context;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct Exposition {
    std::map<std::string, double> samples;
    std::map<std::string, std::string> types;
};

Exposition parse(const std::string& body) {
    Exposition parsed;
    std::istringstream lines(body);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("# TYPE ", 0) == 0) {
            std::istringstream fields(line.substr(7));
            std::string name;
            std::string type;
            fields >> name >> type;
            parsed.types[name] = type;
            continue;
        }
        if (line.empty() || line.front() == '#') { continue; }
        const std::size_t space = line.find(' ');
        if (space == std::string::npos) { continue; }
        parsed.samples[line.substr(0, space)] = std::stod(line.substr(space + 1));
    }
    return parsed;
}

GenerationOutcome outcome(int prompt, std::uint32_t cached, int completion, int reasoning,
                          double prefill_seconds, double decode_seconds, std::uint64_t drafted,
                          std::uint64_t accepted) {
    GenerationOutcome out;
    out.prompt_tokens                       = prompt;
    out.completion_tokens                   = completion;
    out.reasoning_tokens                    = reasoning;
    out.metrics.prefix_cache_hit_tokens     = cached;
    out.metrics.prefill_seconds             = prefill_seconds;
    out.metrics.decode_seconds              = decode_seconds;
    out.metrics.speculative_draft_tokens    = drafted;
    out.metrics.speculative_accepted_tokens = accepted;
    return out;
}

} // namespace

int main() {
    int failures = 0;

    ServeMetrics metrics;
    const Exposition empty = parse(metrics.render(gauges(1, 0, 0)));
    failures += check(empty.samples.at("llamacpp:prompt_tokens_total") == 0.0,
                      "prompt token counter does not start at zero");
    failures += check(empty.samples.at("ninfer:requests_total") == 0.0,
                      "request counter does not start at zero");
    failures += check(empty.types.at("llamacpp:prompt_tokens_total") == "counter",
                      "prompt token family is not declared as a counter");
    failures += check(empty.types.size() == empty.samples.size(),
                      "every exposed sample must carry a TYPE declaration");

    // Cold request: the whole prompt is computed by prefill.
    metrics.record_done(outcome(1000, 0, 200, 40, 0.5, 4.0, 300, 150));
    // Warm request: 900 of 1200 prompt tokens come from a reused prefix, so only the 300
    // recomputed tokens may be billed as prefill work.
    metrics.record_done(outcome(1200, 900, 100, 0, 0.1, 2.0, 150, 75));

    const Exposition values = parse(metrics.render(gauges(1, 0, 0)));
    failures += check(values.samples.at("llamacpp:prompt_tokens_total") == 1300.0,
                      "computed prefill tokens exclude reused prefix tokens");
    failures += check(values.samples.at("llamacpp:prompt_seconds_total") == 0.6,
                      "prefill seconds are not summed");
    failures += check(values.samples.at("llamacpp:tokens_predicted_total") == 300.0,
                      "completion tokens are not summed");
    failures += check(values.samples.at("llamacpp:tokens_predicted_seconds_total") == 6.0,
                      "decode seconds are not summed");
    failures += check(values.samples.at("ninfer:requests_total") == 2.0,
                      "completed requests are not counted");
    failures += check(values.samples.at("ninfer:requests_failed_total") == 0.0,
                      "completed requests must not count as failures");
    failures += check(values.samples.at("ninfer:reasoning_tokens_total") == 40.0,
                      "reasoning tokens are not summed");
    failures += check(values.samples.at("ninfer:prefix_cache_hit_tokens_total") == 900.0,
                      "prefix cache hits are not summed");
    failures += check(values.samples.at("ninfer:draft_tokens_total") == 450.0,
                      "speculative draft tokens are not summed");
    failures += check(values.samples.at("ninfer:draft_accepted_tokens_total") == 225.0,
                      "accepted draft tokens are not summed");

    // A reuse report larger than the prompt must clamp both series, never wrap.
    metrics.record_done(outcome(10, 50, 1, 0, 0.0, 0.1, 0, 0));
    const Exposition clamped = parse(metrics.render(gauges(1, 0, 0)));
    failures += check(clamped.samples.at("llamacpp:prompt_tokens_total") == 1300.0,
                      "prefill token counter underflowed on an oversized reuse report");
    failures += check(clamped.samples.at("ninfer:prefix_cache_hit_tokens_total") == 910.0,
                      "prefix cache counter exceeded the prompt it was reported against");

    RequestFailure failure;
    failure.phase          = RequestFailurePhase::Generation;
    failure.classification = RequestFailureClass::Internal;
    metrics.record_failure(failure);
    metrics.record_rejected();
    RequestFailure cancelled;
    cancelled.phase          = RequestFailurePhase::Transport;
    cancelled.classification = RequestFailureClass::ClientDisconnected;
    metrics.record_failure(cancelled);
    const Exposition terminal = parse(metrics.render(gauges(1, 0, 0)));
    failures += check(terminal.samples.at("ninfer:requests_total") == 6.0,
                      "failed, rejected and cancelled requests are not counted as terminal");
    failures += check(terminal.samples.at("ninfer:requests_failed_total") == 2.0,
                      "failed and rejected requests are not counted as failures");
    failures += check(terminal.samples.at("ninfer:requests_cancelled_total") == 1.0,
                      "a client disconnect is not counted as a cancellation");
    failures += check(terminal.samples.at("llamacpp:tokens_predicted_total") == 301.0,
                      "a failure must not change token accounting");

    // Gauges are the Engine's published occupancy, not a serve-layer recount.
    const Exposition busy = parse(metrics.render(gauges(2, 2, 3)));
    failures += check(busy.samples.at("llamacpp:requests_processing") == 2.0,
                      "processing gauge does not follow the Engine lane occupancy");
    failures += check(busy.samples.at("llamacpp:requests_deferred") == 3.0,
                      "deferred gauge does not follow the Engine ingress queue");
    failures += check(busy.types.at("llamacpp:requests_processing") == "gauge",
                      "occupancy families must be declared as gauges");

    // Two lanes, three accepted requests, one lane still free: the two oldest are resident, the
    // newest is the pending tail, and the free lane reads idle.
    metrics.begin_request(started(7, "openai.chat", 500));
    metrics.begin_request(started(8, "anthropic.messages", 900));
    metrics.begin_request(started(9, "openai.responses", 100));
    const std::vector<InFlightRequest> snapshot = metrics.in_flight_snapshot();
    failures +=
        check(snapshot.size() == 3 && snapshot[0].request_id == 7 && snapshot[2].request_id == 9,
              "in-flight snapshot is not in arrival order");
    failures +=
        check(snapshot[1].prompt_tokens == 900 && snapshot[1].protocol == "anthropic.messages",
              "in-flight snapshot lost request detail");

    const nlohmann::json slots =
        nlohmann::json::parse(render_slots_json(gauges(3, 2, 1), snapshot));
    failures += check(slots.at("slots").size() == 3, "one slot per configured execution lane");
    failures += check(slots.at("slots").at(0).at("state") == "processing" &&
                          slots.at("slots").at(0).at("request_id") == 7 &&
                          slots.at("slots").at(0).at("n_prompt_tokens") == 500,
                      "the oldest accepted request does not occupy the first lane");
    failures += check(slots.at("slots").at(1).at("request_id") == 8,
                      "the second lane does not hold the second oldest request");
    failures += check(slots.at("slots").at(2).at("state") == "idle" &&
                          slots.at("slots").at(2).at("request_id").is_null(),
                      "an unoccupied lane is not reported idle");
    failures += check(slots.at("slots").at(0).at("n_ctx") == 65536 &&
                          slots.at("slots").at(0).at("speculative") == true,
                      "slot lane configuration is not reported");
    failures +=
        check(slots.at("pending").size() == 1 && slots.at("pending").at(0).at("request_id") == 9,
              "the FIFO tail beyond the occupied lanes is not reported as pending");
    failures += check(slots.at("requests_processing") == 2 && slots.at("requests_deferred") == 1,
                      "slot totals disagree with the lane assignment");

    // A terminal outcome releases the entry on every path, including failure.
    metrics.end_request(7);
    metrics.end_request(7);
    metrics.end_request(8);
    metrics.end_request(9);
    failures += check(metrics.in_flight_snapshot().empty(),
                      "a terminal request left an occupied in-flight entry behind");

    // An Engine that reports more resident requests than the serve layer tracks must not index
    // past the snapshot.
    const nlohmann::json drained =
        nlohmann::json::parse(render_slots_json(gauges(2, 2, 0), metrics.in_flight_snapshot()));
    failures += check(drained.at("requests_processing") == 0 &&
                          drained.at("slots").at(0).at("state") == "idle",
                      "lane assignment exceeded the tracked in-flight requests");

    // The persistent prompt-cache families are always exported so a scrape config does not have
    // to know whether --prompt-cache is on; they read zero while the tier is absent.
    const Exposition idle_cache = parse(metrics.render(gauges(1, 0, 0)));
    failures += check(idle_cache.samples.at("ninfer:prompt_cache_lookups_total") == 0.0 &&
                          idle_cache.samples.at("ninfer:prompt_cache_records") == 0.0,
                      "prompt cache families are missing when the disk tier is disabled");

    ninfer::RuntimeStats stats;
    stats.prompt_cache_lookups          = 11;
    stats.prompt_cache_hits             = 7;
    stats.prompt_cache_restores         = 5;
    stats.prompt_cache_restore_failures = 2;
    stats.prompt_cache_restored_bytes   = 4096;
    stats.prompt_cache_spill_requests   = 9;
    stats.prompt_cache_spills           = 6;
    stats.prompt_cache_spills_dropped   = 3;
    stats.prompt_cache_spilled_bytes    = 8192;
    stats.prompt_cache_records          = 4;
    stats.prompt_cache_evictions        = 1;
    stats.prompt_cache_compactions      = 2;
    stats.prompt_cache_live_bytes       = 65536;
    stats.prompt_cache_file_bytes       = 131072;
    ninfer::serve::ServeOptions cache_options;
    cache_options.max_concurrency = 1;
    cache_options.max_context     = 65536;
    const Exposition live_cache =
        parse(metrics.render(ninfer::serve::make_executor_gauges(cache_options, stats)));
    failures +=
        check(live_cache.samples.at("ninfer:prompt_cache_lookups_total") == 11.0 &&
                  live_cache.samples.at("ninfer:prompt_cache_hits_total") == 7.0 &&
                  live_cache.samples.at("ninfer:prompt_cache_restores_total") == 5.0 &&
                  live_cache.samples.at("ninfer:prompt_cache_restore_failures_total") == 2.0 &&
                  live_cache.samples.at("ninfer:prompt_cache_restored_bytes_total") == 4096.0,
              "prompt cache restore counters are not taken from RuntimeStats");
    failures +=
        check(live_cache.samples.at("ninfer:prompt_cache_spill_requests_total") == 9.0 &&
                  live_cache.samples.at("ninfer:prompt_cache_spills_total") == 6.0 &&
                  live_cache.samples.at("ninfer:prompt_cache_spills_dropped_total") == 3.0 &&
                  live_cache.samples.at("ninfer:prompt_cache_spilled_bytes_total") == 8192.0,
              "prompt cache spill counters are not taken from RuntimeStats");
    failures += check(live_cache.samples.at("ninfer:prompt_cache_records") == 4.0 &&
                          live_cache.samples.at("ninfer:prompt_cache_evictions_total") == 1.0 &&
                          live_cache.samples.at("ninfer:prompt_cache_compactions_total") == 2.0 &&
                          live_cache.samples.at("ninfer:prompt_cache_live_bytes") == 65536.0 &&
                          live_cache.samples.at("ninfer:prompt_cache_file_bytes") == 131072.0,
                      "prompt cache store occupancy is not taken from RuntimeStats");
    failures += check(live_cache.types.at("ninfer:prompt_cache_records") == "gauge" &&
                          live_cache.types.at("ninfer:prompt_cache_live_bytes") == "gauge" &&
                          live_cache.types.at("ninfer:prompt_cache_lookups_total") == "counter",
                      "prompt cache occupancy and cumulative families are typed incorrectly");

    if (failures == 0) { std::cout << "serve metrics OK\n"; }
    return failures == 0 ? 0 : 1;
}
