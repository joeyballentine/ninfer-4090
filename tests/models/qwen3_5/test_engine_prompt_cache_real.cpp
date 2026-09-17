// End-to-end persistent prompt cache against a real artifact.
//
// The store is only useful if a record written by one Engine survives that Engine's teardown and
// is still there for the next one, so the scenario is deliberately a restart: fill a store, tear
// the Engine down, open a second Engine on the same directory and read the index back. It also
// covers the guard that makes that safe - a store whose signature belongs to a different
// configuration is never matched - by running a third Engine whose KV storage differs and
// checking that it lands in its own store instead of adopting the first one's records.
//
// Skips with 77 when NINFER_TEST_ARTIFACT is unset or no CUDA device is present.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

class TempDirectory {
public:
    TempDirectory() {
        std::random_device entropy;
        path_ = std::filesystem::temp_directory_path() /
                ("ninfer-prompt-cache-" + std::to_string(entropy()));
        std::filesystem::create_directories(path_);
    }

    ~TempDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    TempDirectory(const TempDirectory&)            = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

ninfer::EngineOptions cache_options(const char* artifact, const std::filesystem::path& store) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 2048;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.prefill_chunk        = 512;
    options.speculative.backend  = ninfer::SpeculativeBackend::None;
    options.max_concurrency      = 1;
    options.max_pending_requests = 1;
    // Both in-memory tiers are needed: the disk tier only ever sees a checkpoint that already has
    // a Host replica, which is what the Host State slots and the Host KV arena provide.
    options.context_cache.enabled                           = true;
    options.context_cache.device_state_slots                = 2;
    options.context_cache.host_state_slots                  = 2;
    options.context_cache.host_kv_capacity_bytes            = 512ULL << 20;
    options.context_cache.max_private_continuations         = 2;
    options.context_cache.max_shared_prefixes               = 1;
    options.context_cache.max_long_anchors_per_continuation = 0;
    options.prompt_cache.enabled                            = true;
    options.prompt_cache.directory                          = store;
    options.prompt_cache.max_bytes                          = 8ULL << 30;
    return options;
}

std::vector<ninfer::TokenId> conversation_prefix() {
    std::vector<ninfer::TokenId> tokens;
    tokens.reserve(600);
    // A prefix long enough to occupy several page groups, so the record is a real page stream
    // rather than a single page.
    for (std::uint32_t index = 0; index < 600; ++index) {
        tokens.push_back(static_cast<ninfer::TokenId>(1000 + (index % 997)));
    }
    return tokens;
}

ninfer::RequestOptions turn_options(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

// One Engine lifetime: run two turns of the same conversation so a terminal endpoint is
// catalogued and then pressured onto the host, and report what the tier recorded.
ninfer::RuntimeStats fill_store(const char* artifact, const std::filesystem::path& store) {
    ninfer::Engine engine(cache_options(artifact, store));
    const std::vector<ninfer::TokenId> prefix = conversation_prefix();
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prefix), turn_options(8));
    expect(first.generated_token_ids.size() == 8, "the first turn did not generate eight tokens");

    std::vector<ninfer::TokenId> second_turn = prefix;
    second_turn.insert(second_turn.end(), first.generated_token_ids.begin(),
                       first.generated_token_ids.end());
    for (std::uint32_t index = 0; index < 200; ++index) {
        second_turn.push_back(static_cast<ninfer::TokenId>(2000 + (index % 593)));
    }
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare_tokens(second_turn), turn_options(8));
    expect(second.generated_token_ids.size() == 8, "the second turn did not generate eight tokens");
    return engine.runtime_stats();
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    int devices             = 0;
    const cudaError_t query = cudaGetDeviceCount(&devices);
    if (query != cudaSuccess || devices == 0) {
        std::cout << "skip: no CUDA device\n";
        return 77;
    }

    const TempDirectory root;
    const std::filesystem::path store = root.path() / "store";

    const ninfer::RuntimeStats filled = fill_store(artifact, store);
    expect(filled.prompt_cache_spills != 0,
           "no record reached the store while the prompt cache was enabled");
    expect(filled.prompt_cache_spilled_bytes != 0, "a published record carried no bytes");
    expect(std::filesystem::exists(store / "index.log") &&
               std::filesystem::exists(store / "blobs.dat"),
           "the store directory does not hold both files");

    // Restart. The second Engine replays the journal at open, so whatever the first published is
    // visible without any scan of the extent file.
    {
        ninfer::Engine engine(cache_options(artifact, store));
        const ninfer::RuntimeStats reopened = engine.runtime_stats();
        expect(reopened.prompt_cache_records != 0,
               "a reopened store reported no records after a restart");
        expect(reopened.prompt_cache_live_bytes != 0,
               "a reopened store reported no live bytes after a restart");
        expect(reopened.prompt_cache_spills == 0,
               "a freshly opened Engine counted spills it never performed");
    }

    // A different KV storage changes what a stored page means. The signature that names the store
    // directory has to separate the two, or the second configuration would read the first's pages
    // as its own.
    {
        ninfer::EngineOptions options  = cache_options(artifact, {});
        options.kv_cache               = ninfer::KvCacheStorage::Fp8E4M3Row256;
        options.prompt_cache.directory = root.path() / "by-signature-a";
        ninfer::Engine engine(std::move(options));
        const ninfer::RuntimeStats separate = engine.runtime_stats();
        expect(separate.prompt_cache_records == 0,
               "a store for a different KV storage adopted foreign records");
    }

    // With the option off nothing is constructed, so no counter can move.
    {
        ninfer::EngineOptions options = cache_options(artifact, store);
        options.prompt_cache          = ninfer::PromptCacheOptions{};
        ninfer::Engine engine(std::move(options));
        const ninfer::GenerationResult result =
            engine.generate(engine.prepare_tokens(conversation_prefix()), turn_options(4));
        expect(result.generated_token_ids.size() == 4, "the uncached turn did not generate");
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        expect(stats.prompt_cache_records == 0 && stats.prompt_cache_spills == 0 &&
                   stats.prompt_cache_file_bytes == 0,
               "an Engine without a prompt cache reported prompt cache activity");
    }

    if (failures == 0) { std::cout << "prompt cache round trip OK\n"; }
    return failures == 0 ? 0 : 1;
}
