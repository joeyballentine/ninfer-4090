#pragma once

// Identity of a persistent prompt-cache store.
//
// A disk record is a State image plus opaque KV page images. Reading one back into a Program whose
// weights, KV schedule or resolved geometry differ produces a state that is silently wrong rather
// than obviously broken, so `ContextDiskStore` refuses any store whose header signature is not
// byte-identical to the running configuration. This file composes that signature.
//
// The composition is deliberately Program-owned: only the model knows which facts change the
// meaning of a page. The Engine supplies the artifact half and the Program adds the rest.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5 {

// Every fact that changes what a stored page or image means. Adding a field is a store-format
// change by construction: the signature moves and old records stop matching.
struct ContextCacheSignatureFacts {
    // `LoadSummary::prefill_signature`: architecture, config interpretation and the exact bound
    // weight inventory, including each part's stored format. It is long, so only its digest enters
    // the signature.
    std::string_view artifact_identity;
    // KV storage per layer. `identity_tag` is the same value the in-memory prefix identity uses,
    // so a schedule that cannot read another schedule's pages also cannot match its records.
    KvCacheSchedule kv_storage;
    // The backend owns a second KV pool and, for DFlash, part of the State image.
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    ProposalHead proposal_head             = ProposalHead::Full;
    // Resolved geometry. Page tokens and the host page stride fix the byte layout of one record
    // page; the plane count fixes how many independently laid-out planes that stride covers, which
    // is what a per-layer schedule varies.
    std::uint32_t max_context         = 0;
    std::uint32_t kv_capacity         = 0;
    std::uint32_t main_page_tokens    = 0;
    std::uint32_t main_plane_count    = 0;
    std::size_t main_page_bytes       = 0;
    std::uint32_t backend_page_tokens = 0;
    std::uint32_t backend_plane_count = 0;
    std::size_t backend_page_bytes    = 0;
    std::size_t state_image_bytes     = 0;
    std::uint32_t vision_enabled      = 0;
};

// 64-bit FNV-1a, rendered as 16 lowercase hex digits. Exposed because the composition is tested
// against it rather than against a hard-coded digest of a long artifact signature.
[[nodiscard]] std::string context_cache_identity_digest(std::string_view text) noexcept;

// One line, filesystem-safe, stable across runs of the same configuration. The leading token is a
// format version: changing how any later field is spelled must bump it, because two different
// compositions that happen to collide would match each other's records.
[[nodiscard]] std::string context_cache_signature(const ContextCacheSignatureFacts& facts);

} // namespace ninfer::models::qwen3_5
