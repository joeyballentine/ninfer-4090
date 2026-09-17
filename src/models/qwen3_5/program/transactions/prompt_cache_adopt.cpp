// The Program side of hook (b) of the persistent prompt cache: the transaction that turns a
// restored disk record into a catalogued continuation. The physical transfers it drives belong to
// `prompt_cache_port.*`; this file owns the logical state change, which is why it sits with the
// Program's other transactions.

#include "models/qwen3_5/program/prompt_cache_port.h"

#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/program_impl.h"

#include <exception>
#include <optional>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// Turns the record the restore just placed in Host memory into a catalogued continuation, so the
// planner prices it exactly like a checkpoint this process produced and only the uncovered suffix
// is prefilled. Every step below is an existing Program entry point; adoption adds no new
// physical authority, which is why the invariants it has to respect are the ones those entry
// points already enforce:
//
//   * the KV pages come from a real Device page reservation, so `Used_r(S)` counts them the
//     moment they are materialized - `physical_occupancy` reads the pools, not a side table;
//   * the frontier is committed with `commit_frontier`, so committed columns and content epochs
//     are what a Host replica may be attached against;
//   * the Host extent is published with the record's own arena allocation, so the adopted
//     checkpoint is `Both`-resident without a second copy and is immediately re-spillable;
//   * the State image becomes a Host-only `CheckpointImmutable` object, which is the same shape
//     pressure leaves behind when it demotes an endpoint, so its H2D restore is already priced;
//   * nothing resident is evicted or retargeted: a missing free continuation slot, a missing
//     free execution row or a Device page reservation that would not fit declines instead.
qwen3_5::PromptCacheAdoptionResult ProgramImpl::adopt_prompt_cache_record() {
    using Adoption = qwen3_5::PromptCacheAdoption;
    qwen3_5::PromptCacheAdoptionResult out;
    if (!prompt_cache_port_) { return out; }
    std::optional<PromptCacheRestoredRecord> restored = prompt_cache_port_->take_restored_record();
    if (!restored) { return out; }

    // The restore owns a Host State slot and a filled Host KV allocation. Each is handed to a
    // store exactly once; whatever is still held here when this returns goes back to its pool.
    bool owns_pages            = true;
    bool owns_slot             = true;
    const auto release_untaken = [&]() noexcept {
        if (owns_pages && restored->pages.valid()) { (void)restored->pages.release(); }
        if (owns_slot) { (void)host_state_images->release(restored->state_slot); }
        owns_pages = false;
        owns_slot  = false;
    };
    // Sets the reason and returns the restore's resources; the caller then returns `out`, which
    // is the function's named return value and so is never copied.
    const auto decline = [&](Adoption status) noexcept {
        release_untaken();
        out.status = status;
    };

    const PromptCacheRecordHeader& header = restored->header;
    if (has_context_transaction() || pending_transaction_) {
        decline(Adoption::ProgramBusy);
        return out;
    }
    // A second KV pool would have to be assembled and committed alongside the main one, and with
    // MTP a reused checkpoint is only materializable when its draft KV came back too. Neither is
    // implemented, so a record from a Program with a speculative backend declines here.
    if (backend_kv_pages != nullptr || speculative_backend != SpeculativeBackend::None ||
        header.backend_pages != 0 || header.backend_frontier != 0) {
        decline(Adoption::BackendPoolUnsupported);
        return out;
    }
    // Only a session endpoint is adopted. Its frontier is the owner's execution frontier, which
    // is what makes the record's rope delta and execution frontier describe the same position the
    // committed KV and State image do; a rewrite checkpoint or long anchor sits behind the
    // execution frontier and needs its own continuation shape.
    if (header.checkpoint_kind !=
            static_cast<std::uint32_t>(runtime::CheckpointKind::SessionEndpoint) ||
        header.checkpoint_ordinal != 0 || header.frontier == 0 || header.frontier > capacity ||
        header.frontier != header.execution_frontier || header.frontier != header.text_kv_valid ||
        header.mtp_kv_valid != 0 || header.dflash_context_frontier != 0) {
        decline(Adoption::UnsupportedRecord);
        return out;
    }
    const std::uint32_t frontier   = header.frontier;
    const std::uint32_t main_pages = kv_pages_for_frontier(frontier);
    if (main_pages == 0 || main_pages != header.main_pages ||
        restored->pages.page_count() != main_pages) {
        decline(Adoption::UnsupportedRecord);
        return out;
    }
    // Without the exact identity the continuation could be shortlisted and never verified, which
    // is the one way a disk hit could return someone else's tokens.
    if (restored->ledger.size() != frontier || restored->identity.size() != frontier ||
        restored->digests.size() != frontier) {
        decline(Adoption::MissingIdentity);
        return out;
    }

    // Pressure rule: the Device pages this checkpoint needs have to fit beside everything already
    // resident. Evicting a resident owner to make room would trade a proven in-memory source for
    // a restored one.
    detail::PhysicalResources peak;
    peak.device.main_kv_pages = main_pages;
    if (!physical_peak_fits(peak)) {
        decline(Adoption::DevicePagesUnavailable);
        return out;
    }

    // An execution row is needed only while the pages are being committed:
    // `ensure_mapped_to_tokens` and `commit_frontier` are active-address-space operations, and rows
    // are one per lane.
    std::optional<std::int32_t> row;
    for (std::uint32_t candidate = 0; candidate < text_kv_addresses->execution_row_count();
         ++candidate) {
        if (text_kv_addresses->execution_row_free(static_cast<std::int32_t>(candidate))) {
            row = static_cast<std::int32_t>(candidate);
            break;
        }
    }
    const std::optional<std::uint32_t> index =
        row ? allocate_continuation_slot() : std::optional<std::uint32_t>();
    if (!row || !index) {
        decline(Adoption::NoLogicalSlot);
        return out;
    }

    SequenceState& sequence = continuation_states[*index];
    try {
        out.summary.long_anchors.reserve(
            context_cache.max_long_anchors_per_continuation.value_or(0));
        sequence.long_anchors.clear();
        sequence.shared_prefix_references.clear();
        sequence.rewrite_state.reset();
        sequence.reserved_state.reset();
        sequence.rewrite_checkpoint = {};

        const std::optional<KVAddressSpaceHandle> text =
            text_kv_addresses->create_active(main_pages, *row);
        if (!text) { throw std::bad_alloc(); }
        sequence.kv.emplace(SequenceKVBundle{.text = *text});
        text_kv_addresses->ensure_mapped_to_tokens(*text, frontier, device.stream);

        std::vector<DeviceKVPageHandle> destinations(main_pages);
        std::vector<LogicalKVPageHandle> membership(main_pages);
        for (std::uint32_t page = 0; page < main_pages; ++page) {
            destinations[page] = text_kv_addresses->physical_page(*text, page);
            membership[page]   = text_kv_addresses->logical_page(*text, page);
        }
        text_kv_pages->physical_pool().copy_from_host(host_kv_arena->view(restored->pages),
                                                      destinations, device.stream);
        // The copy has to be complete before the frontier is committed: from that point on the
        // pages are a checkpoint other requests may read.
        device.synchronize();
        text_kv_addresses->commit_frontier(*text, frontier);
        text_kv_addresses->set_checkpoint_requirement(*text, frontier);
        text_kv_addresses->deactivate(*text);

        // Publishing the record's own allocation as the Host extent is what makes the adopted
        // checkpoint Host-resident without copying a single page twice. It has to happen after
        // the deactivation, because a page with a writer reference is not a pinnable source.
        std::optional<HostKVExtentReservation> extent = host_kv_extents->prepare_adopted(
            *text_kv_pages, membership, std::move(restored->pages));
        if (!extent) { throw std::bad_alloc(); }
        owns_pages = false;
        (void)host_kv_extents->publish(std::move(*extent));

        const std::optional<StateImageHandle> state =
            state_store->adopt_host_checkpoint(restored->state_slot);
        if (!state) { throw std::bad_alloc(); }
        owns_slot      = false;
        sequence.state = ActiveStateBinding{.read = *state, .write = *state};

        sequence.lane               = 0;
        sequence.execution_frontier = frontier;
        sequence.ledger_frontier    = frontier;
        sequence.ledger             = std::move(restored->ledger);
        sequence.prefix_identity.swap(restored->identity);
        sequence.prefix_digests.swap(restored->digests);
        sequence.rope_delta              = header.rope_delta;
        sequence.text_kv_valid           = frontier;
        sequence.mtp_kv_valid            = 0;
        sequence.dflash_context_frontier = 0;
        sequence.mtp_draft_count         = 0;
        // No hidden tail came back with the record; only the MTP bridge reads it, and this
        // Program has no backend.
        sequence.tail_hidden_valid  = false;
        sequence.endpoint_valid     = true;
        sequence.rebuild_work       = runtime::make_prefill_work(0, frontier, header.vision_items,
                                                                 header.vision_patches, prefill_chunk);
        sequence.rebuild_tail_begin = 0;
        refresh_state_views(sequence);
        continuation_slots[*index].role = ContinuationSlotRole::Catalogued;
        populate_continuation_summary(sequence, out.summary);
        out.summary.active_references = 0;
    } catch (const std::exception&) {
        // Everything already handed to a store is released with the slot; the rest goes back to
        // its pool. The slot itself returns to Free, so a failed adoption costs nothing but the
        // I/O the restore already spent.
        release_untaken();
        release_continuation_slot_best_effort(*index);
        out.status = Adoption::TransactionFailed;
        return out;
    }

    out.continuation.emplace(
        ContractAccess::make_continuation(this, *index, continuation_slots[*index].generation));
    out.status = Adoption::Adopted;
    advance_resource_revision();
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
