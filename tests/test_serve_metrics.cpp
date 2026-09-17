#include "serve/serve_metrics.h"

#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace {

using ninfer::serve::GenerationOutcome;
using ninfer::serve::RequestFailure;
using ninfer::serve::RequestFailureClass;
using ninfer::serve::RequestFailurePhase;
using ninfer::serve::ServeMetrics;

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
    const Exposition empty = parse(metrics.render());
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

    const Exposition values = parse(metrics.render());
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
    const Exposition clamped = parse(metrics.render());
    failures += check(clamped.samples.at("llamacpp:prompt_tokens_total") == 1300.0,
                      "prefill token counter underflowed on an oversized reuse report");
    failures += check(clamped.samples.at("ninfer:prefix_cache_hit_tokens_total") == 910.0,
                      "prefix cache counter exceeded the prompt it was reported against");

    RequestFailure failure;
    failure.phase          = RequestFailurePhase::Generation;
    failure.classification = RequestFailureClass::Internal;
    metrics.record_failure(failure);
    metrics.record_rejected();
    const Exposition terminal = parse(metrics.render());
    failures += check(terminal.samples.at("ninfer:requests_total") == 5.0,
                      "failed and rejected requests are not counted as terminal");
    failures += check(terminal.samples.at("ninfer:requests_failed_total") == 2.0,
                      "failed and rejected requests are not counted as failures");
    failures += check(terminal.samples.at("llamacpp:tokens_predicted_total") == 301.0,
                      "a failure must not change token accounting");

    if (failures == 0) { std::cout << "serve metrics OK\n"; }
    return failures == 0 ? 0 : 1;
}
