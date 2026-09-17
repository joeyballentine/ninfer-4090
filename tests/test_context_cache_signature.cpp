// Composition of the persistent prompt-cache store signature.
//
// The store refuses records whose header signature differs, so this test is the guard that every
// fact which changes what a stored page means actually reaches the signature. A field that stops
// being composed in would let a store written by one configuration be restored into another.

#include "models/qwen3_5/program/context_cache_signature.h"

#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::KvCacheSchedule;
using ninfer::KvCacheStorage;
using ninfer::ProposalHead;
using ninfer::SpeculativeBackend;
using ninfer::models::qwen3_5::context_cache_identity_digest;
using ninfer::models::qwen3_5::context_cache_signature;
using ninfer::models::qwen3_5::ContextCacheSignatureFacts;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ContextCacheSignatureFacts baseline() {
    ContextCacheSignatureFacts facts;
    facts.artifact_identity   = "{\"implementation\":\"qwen3_5-prefill-1\"}";
    facts.kv_storage          = KvCacheSchedule(KvCacheStorage::BFloat16);
    facts.speculative_backend = SpeculativeBackend::Mtp;
    facts.proposal_head       = ProposalHead::Full;
    facts.max_context         = 65536;
    facts.kv_capacity         = 65536;
    facts.main_page_tokens    = 64;
    facts.main_plane_count    = 96;
    facts.main_page_bytes     = 1U << 20;
    facts.backend_page_tokens = 64;
    facts.backend_plane_count = 2;
    facts.backend_page_bytes  = 16384;
    facts.state_image_bytes   = 4U << 20;
    facts.vision_enabled      = 0;
    return facts;
}

} // namespace

int main() {
    int failures = 0;

    const ContextCacheSignatureFacts base = baseline();
    const std::string signature           = context_cache_signature(base);

    failures += check(signature.rfind("q35-ctx1.", 0) == 0,
                      "the signature does not start with its format version");
    failures += check(signature == context_cache_signature(baseline()),
                      "the signature is not stable for identical facts");

    // A directory name is derived from this string; every character must survive that reduction
    // unchanged, or two signatures could name the same directory.
    for (const char character : signature) {
        const bool plain = (character >= '0' && character <= '9') ||
                           (character >= 'a' && character <= 'z') ||
                           (character >= 'A' && character <= 'Z') || character == '-' ||
                           character == '.' || character == '_';
        failures += check(plain, "the signature contains a character a directory name cannot hold");
        if (!plain) { break; }
    }

    // The artifact half is a digest, so a long weight inventory cannot dominate the string, but a
    // single changed byte must still move it.
    failures += check(signature.find(context_cache_identity_digest(base.artifact_identity)) !=
                          std::string::npos,
                      "the artifact identity digest is not part of the signature");
    failures += check(context_cache_identity_digest("a") != context_cache_identity_digest("b"),
                      "the artifact identity digest ignores its input");
    failures += check(context_cache_identity_digest("a").size() == 16,
                      "the artifact identity digest is not a fixed 16 hex digits");

    // Every fact is load bearing: changing any one of them must produce a different store.
    std::vector<std::pair<const char*, ContextCacheSignatureFacts>> variants;
    const auto vary = [&](const char* name, auto&& mutate) {
        ContextCacheSignatureFacts facts = baseline();
        mutate(facts);
        variants.emplace_back(name, facts);
    };
    vary("artifact", [](ContextCacheSignatureFacts& f) { f.artifact_identity = "other artifact"; });
    vary("kv storage", [](ContextCacheSignatureFacts& f) {
        f.kv_storage = KvCacheSchedule(KvCacheStorage::Int8Group64);
    });
    vary("kv schedule boundary", [](ContextCacheSignatureFacts& f) {
        f.kv_storage = KvCacheSchedule(KvCacheStorage::BFloat16, 8, KvCacheStorage::Int8Group64);
    });
    vary("speculative backend",
         [](ContextCacheSignatureFacts& f) { f.speculative_backend = SpeculativeBackend::DFlash; });
    vary("proposal head",
         [](ContextCacheSignatureFacts& f) { f.proposal_head = ProposalHead::Optimized; });
    vary("max context", [](ContextCacheSignatureFacts& f) { f.max_context = 32768; });
    vary("kv capacity", [](ContextCacheSignatureFacts& f) { f.kv_capacity = 32768; });
    vary("main page tokens", [](ContextCacheSignatureFacts& f) { f.main_page_tokens = 32; });
    vary("main plane count", [](ContextCacheSignatureFacts& f) { f.main_plane_count = 64; });
    vary("main page bytes", [](ContextCacheSignatureFacts& f) { f.main_page_bytes = 1U << 19; });
    vary("backend page tokens", [](ContextCacheSignatureFacts& f) { f.backend_page_tokens = 32; });
    vary("backend plane count", [](ContextCacheSignatureFacts& f) { f.backend_plane_count = 4; });
    vary("backend page bytes", [](ContextCacheSignatureFacts& f) { f.backend_page_bytes = 8192; });
    vary("state image bytes",
         [](ContextCacheSignatureFacts& f) { f.state_image_bytes = 2U << 20; });
    vary("vision", [](ContextCacheSignatureFacts& f) { f.vision_enabled = 1; });

    std::set<std::string> seen{signature};
    for (const auto& [name, facts] : variants) {
        const std::string varied = context_cache_signature(facts);
        failures += check(varied != signature, name);
        failures += check(seen.insert(varied).second, name);
    }

    // A two-tier schedule and a uniform schedule of the same tail kind read different pages, and
    // the identity tag is what separates them in memory as well.
    const KvCacheSchedule uniform(KvCacheStorage::Int8Group64);
    const KvCacheSchedule tiered(KvCacheStorage::BFloat16, 8, KvCacheStorage::Int8Group64);
    failures += check(uniform.identity_tag() != tiered.identity_tag(),
                      "the KV schedule identity tag does not separate a two-tier schedule");
    ContextCacheSignatureFacts uniform_facts = baseline();
    uniform_facts.kv_storage                 = uniform;
    ContextCacheSignatureFacts tiered_facts  = baseline();
    tiered_facts.kv_storage                  = tiered;
    failures +=
        check(context_cache_signature(uniform_facts) != context_cache_signature(tiered_facts),
              "a two-tier schedule shares a store with its uniform tail kind");

    if (failures == 0) { std::cout << "context cache signature OK\n"; }
    return failures == 0 ? 0 : 1;
}
