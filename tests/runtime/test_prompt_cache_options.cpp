// Engine-level validation and store configuration for the persistent prompt cache.
//
// Both halves run without a GPU: `normalize_engine_options` is pure option arithmetic, and
// `resolve_prompt_cache_config` only derives paths and limits.

#include "runtime/engine/context_cache/context_disk_tier.h"
#include "runtime/engine/model_instance.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::EngineOptions generation_options() {
    ninfer::EngineOptions options;
    options.purpose               = ninfer::EnginePurpose::Generation;
    options.artifact_path         = "/models/qwen3_6_27b.ninfer";
    options.max_context           = 4096;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.max_concurrency       = 2;
    options.context_cache.enabled = true;
    options.prompt_cache.enabled  = true;
    return options;
}

template <class Call>
bool rejected(Call&& call) {
    try {
        (void)call();
    } catch (const std::invalid_argument&) { return true; } catch (...) {
        return false;
    }
    return false;
}

} // namespace

int main() {
    int failures = 0;

    const ninfer::EngineOptions normalized =
        ninfer::runtime::normalize_engine_options(generation_options());
    failures += check(normalized.prompt_cache.enabled,
                      "a generation Engine with a context cache lost its prompt cache");
    failures += check(normalized.context_cache.host_state_slots != 0,
                      "the prompt cache was accepted without a Host State restore destination");

    // The disk tier is the third tier of the context cache. Without the first two there is nothing
    // to spill and nowhere to restore into, so the combination is rejected rather than silently
    // producing a store that never fills.
    failures += check(rejected([] {
                          ninfer::EngineOptions options = generation_options();
                          options.context_cache.enabled = false;
                          return ninfer::runtime::normalize_engine_options(options);
                      }),
                      "the prompt cache was accepted with the context cache disabled");
    failures += check(rejected([] {
                          ninfer::EngineOptions options          = generation_options();
                          options.context_cache.host_state_slots = 0;
                          return ninfer::runtime::normalize_engine_options(options);
                      }),
                      "the prompt cache was accepted with no Host State slot");
    failures += check(rejected([] {
                          ninfer::EngineOptions options  = generation_options();
                          options.prompt_cache.max_bytes = 0;
                          return ninfer::runtime::normalize_engine_options(options);
                      }),
                      "the prompt cache was accepted with a zero size cap");

    // Scoring publishes no checkpoint, so the purpose clears the tier instead of rejecting it.
    ninfer::EngineOptions scoring = generation_options();
    scoring.purpose               = ninfer::EnginePurpose::CausalScoring;
    const ninfer::EngineOptions scoring_normalized =
        ninfer::runtime::normalize_engine_options(scoring);
    failures += check(!scoring_normalized.prompt_cache.enabled,
                      "a scoring Engine kept a prompt cache it can never fill");

    // Store configuration. An empty directory places the store beside the artifact, under a
    // directory named after the signature, so two configurations of one artifact never share it.
    const ninfer::runtime::ContextDiskStoreConfig derived =
        ninfer::runtime::resolve_prompt_cache_config(normalized, "q35-ctx1.abc.kvbf16");
    failures += check(derived.signature == "q35-ctx1.abc.kvbf16",
                      "the resolved store did not keep the signature verbatim");
    failures += check(derived.directory.parent_path().filename() == ".ninfer-cache" &&
                          derived.directory.parent_path().parent_path() == "/models",
                      "the default store is not placed beside the artifact");
    failures += check(derived.directory.filename().string().find("q35-ctx1.abc.kvbf16") == 0,
                      "the default store directory is not named after the signature");
    failures += check(derived.max_bytes == normalized.prompt_cache.max_bytes,
                      "the resolved store did not take the configured size cap");

    ninfer::EngineOptions explicit_dir  = normalized;
    explicit_dir.prompt_cache.directory = "/var/cache/ninfer";
    explicit_dir.prompt_cache.max_bytes = 4096;
    const ninfer::runtime::ContextDiskStoreConfig chosen =
        ninfer::runtime::resolve_prompt_cache_config(explicit_dir, "signature");
    failures += check(chosen.directory == "/var/cache/ninfer",
                      "an explicit prompt cache directory was not used as given");
    failures += check(chosen.max_bytes == 4096, "an explicit size cap was not used as given");

    // Two configurations of one artifact share a long signature prefix. The directory name is
    // bounded, so it must carry a digest of the whole signature rather than a truncation, or the
    // two stores would land in the same directory and reject and reset each other in turn.
    const std::string long_prefix(120, 'x');
    const ninfer::runtime::ContextDiskStoreConfig first =
        ninfer::runtime::resolve_prompt_cache_config(normalized, long_prefix + ".a");
    const ninfer::runtime::ContextDiskStoreConfig second =
        ninfer::runtime::resolve_prompt_cache_config(normalized, long_prefix + ".b");
    failures += check(first.directory != second.directory,
                      "two long signatures sharing a prefix share a store directory");
    failures += check(first.directory.filename().string().size() < 120,
                      "the store directory name is not bounded");

    // A store with no signature could match anything, which is the one failure mode the header
    // signature exists to prevent.
    failures += check(
        rejected([&] { return ninfer::runtime::resolve_prompt_cache_config(normalized, ""); }),
        "an empty signature was accepted for a store");

    if (failures == 0) { std::cout << "prompt cache options OK\n"; }
    return failures == 0 ? 0 : 1;
}
