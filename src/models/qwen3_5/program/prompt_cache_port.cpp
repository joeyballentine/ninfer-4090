#include "models/qwen3_5/program/prompt_cache_port.h"

#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <utility>

namespace ninfer::models::qwen3_5::detail {
namespace {

// Upper bound on the bytes one spill or restore batch holds. A 77k-token record is on the order
// of a thousand page groups; staging all of them at once is the allocation that made the donor
// fork fail on a 24 GB card, so the record is streamed in batches of at most this size.
constexpr std::size_t kTransferBatchBytes  = 64ULL << 20;
constexpr std::uint32_t kMaximumBatchPages = 256;
// The Engine offers at most `queue_depth` spills before the tier starts dropping them. Holding a
// few more snapshots than that costs nothing but keeps a mispaired offer from pinning the whole
// catalog.
constexpr std::size_t kMaximumOutstandingCaptures = 16;

[[nodiscard]] std::uint32_t record_pages_for_stride(std::uint32_t payload_bytes,
                                                    std::uint32_t page_bytes) noexcept {
    if (payload_bytes == 0 || page_bytes == 0) { return 0; }
    return 1U + (payload_bytes - 1U) / page_bytes;
}

} // namespace

PromptCacheTransferPort::PromptCacheTransferPort(ProgramImpl& program) : program_(program) {
    if (!program_.host_kv_extents || !program_.text_kv_pages || !program_.host_state_images ||
        !program_.state_store || !program_.host_kv_arena) {
        throw std::logic_error("the prompt cache port requires the Host context tier");
    }
    page_bytes_         = static_cast<std::uint32_t>(program_.text_host_kv_page_stride);
    backend_page_bytes_ = program_.backend_kv_pages
                              ? static_cast<std::uint32_t>(program_.backend_host_kv_page_stride)
                              : 0U;
    if (page_bytes_ == 0) { throw std::logic_error("the main KV pool has no Host page stride"); }
    backend_record_pages_ = record_pages_for_stride(backend_page_bytes_, page_bytes_);
    batch_pages_          = static_cast<std::uint32_t>(
        std::clamp<std::size_t>(kTransferBatchBytes / page_bytes_, 1, kMaximumBatchPages));
    state_image_bytes_ =
        static_cast<std::uint32_t>(program_.host_state_images->layout().image_bytes);
    capture_staging_.resize(static_cast<std::size_t>(batch_pages_) * page_bytes_);
}

PromptCacheTransferPort::~PromptCacheTransferPort() {
    const std::lock_guard<std::mutex> guard(mutex_);
    for (auto& entry : snapshots_) { release_snapshot(entry.second); }
    snapshots_.clear();
    release_restore();
}

std::size_t PromptCacheTransferPort::outstanding_captures() const noexcept {
    const std::lock_guard<std::mutex> guard(mutex_);
    return snapshots_.size();
}

bool PromptCacheTransferPort::collect_pages(
    KVAddressSpaceStore& addresses, LogicalKVPageStore& pages, KVAddressSpaceHandle address,
    std::uint32_t frontier, std::uint32_t stride, std::vector<LogicalKVPageHandle>& pinned,
    std::vector<PromptCacheCaptureSnapshot::PageSource>& sources) {
    const std::uint32_t required = kv_pages_for_frontier(frontier);
    if (required == 0) { return true; }
    if (stride == 0 || required > addresses.mapped_pages(address)) { return false; }
    const std::uint32_t record_pages = record_pages_for_stride(stride, page_bytes_);
    for (std::uint32_t page = 0; page < required; ++page) {
        const LogicalKVPageHandle logical = addresses.logical_page(address, page);
        if (!pages.can_pin_host_source(logical)) { return false; }
        const std::uint32_t begin = page * static_cast<std::uint32_t>(kPagedKVPageSize);
        const std::uint32_t columns =
            std::min(static_cast<std::uint32_t>(kPagedKVPageSize), frontier - begin);
        // A destructive rewrite leaves a Host replica whose coverage no longer matches the
        // checkpoint. Capturing it would store bytes the frontier does not license.
        if (columns != pages.committed_columns(logical)) { return false; }
        const HostKVPageReplica replica        = pages.host_replica(logical);
        const HostKVAllocationConstView extent = program_.host_kv_extents->view(replica.extent);
        if (extent.layout().page_stride != stride || replica.page_offset >= extent.page_count()) {
            return false;
        }
        const std::byte* source =
            extent.data() + static_cast<std::size_t>(replica.page_offset) * stride;
        pages.pin_source(logical);
        pinned.push_back(logical);
        for (std::uint32_t slice = 0; slice < record_pages; ++slice) {
            const std::uint32_t offset = slice * page_bytes_;
            sources.push_back(PromptCacheCaptureSnapshot::PageSource{
                .data  = source + offset,
                .bytes = std::min(page_bytes_, stride - offset),
            });
        }
    }
    return true;
}

// Concatenates the three identity blobs a record carries behind its State image. A missing piece
// makes the whole set empty: a half-described record would be refused by adoption anyway, and
// storing it would only invite a partial read.
PromptCacheTransferPort::CaptureMetadata PromptCacheTransferPort::build_capture_metadata(
    std::span<const TokenId> ledger, const qwen3_5::detail::ResidentPrefixIdentity* identity,
    const qwen3_5::detail::PrefixShortlistDigests* digests, std::uint32_t frontier) {
    PromptCacheTransferPort::CaptureMetadata out;
    if (identity == nullptr || frontier == 0 || ledger.size() < frontier ||
        identity->size() < frontier) {
        return {};
    }
    qwen3_5::detail::ResidentPrefixIdentity truncated_identity = *identity;
    truncated_identity.truncate(frontier);
    const std::vector<std::byte> identity_bytes = truncated_identity.encode();
    std::vector<std::byte> digest_bytes;
    if (digests != nullptr && digests->size() >= frontier) {
        qwen3_5::detail::PrefixShortlistDigests truncated_digests = *digests;
        truncated_digests.truncate(frontier);
        digest_bytes = truncated_digests.encode();
    }
    const std::size_t ledger_bytes = static_cast<std::size_t>(frontier) * sizeof(TokenId);
    const std::size_t total        = ledger_bytes + identity_bytes.size() + digest_bytes.size();
    if (total > std::numeric_limits<std::uint32_t>::max()) { return {}; }
    out.blob.resize(total);
    std::memcpy(out.blob.data(), ledger.data(), ledger_bytes);
    std::memcpy(out.blob.data() + ledger_bytes, identity_bytes.data(), identity_bytes.size());
    std::memcpy(out.blob.data() + ledger_bytes + identity_bytes.size(), digest_bytes.data(),
                digest_bytes.size());
    out.ledger_bytes   = static_cast<std::uint32_t>(ledger_bytes);
    out.identity_bytes = static_cast<std::uint32_t>(identity_bytes.size());
    out.digest_bytes   = static_cast<std::uint32_t>(digest_bytes.size());
    const qwen3_5::detail::VisionPrefixExtent vision = truncated_identity.vision_extent();
    out.vision_items                                 = vision.items;
    out.vision_patches                               = vision.patches;
    return out;
}

std::optional<std::uint64_t> PromptCacheTransferPort::build_snapshot(SnapshotRequest&& request) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (snapshots_.size() >= kMaximumOutstandingCaptures) { return std::nullopt; }
    if (request.frontier == 0 || !program_.state_store->can_pin_host_source(request.state)) {
        return std::nullopt;
    }

    PromptCacheCaptureSnapshot snapshot;
    const std::uint32_t main_pages    = kv_pages_for_frontier(request.frontier);
    const std::uint32_t backend_pages = kv_pages_for_frontier(request.backend_frontier);
    if (main_pages == 0) { return std::nullopt; }
    snapshot.sources.reserve(static_cast<std::size_t>(main_pages) +
                             static_cast<std::size_t>(backend_pages) * backend_record_pages_);
    // Anything that throws here has to unwind the pins already taken, or the checkpoint stays
    // pinned for the rest of the process and the planner can never move or evict it again.
    const std::byte* state_image = nullptr;
    try {
        const bool collected =
            collect_pages(*program_.text_kv_addresses, *program_.text_kv_pages, request.text,
                          request.frontier, page_bytes_, snapshot.pinned_text_pages,
                          snapshot.sources) &&
            (backend_pages == 0 ||
             (request.backend != nullptr && program_.backend_kv_pages != nullptr &&
              collect_pages(*program_.backend_kv_addresses, *program_.backend_kv_pages,
                            *request.backend, request.backend_frontier, backend_page_bytes_,
                            snapshot.pinned_backend_pages, snapshot.sources)));
        if (collected) {
            const qwen3_5::HostStateImageConstView state =
                program_.state_store->host_view(request.state);
            if (state.data != nullptr && state.layout != nullptr &&
                state.layout->image_bytes == state_image_bytes_) {
                program_.state_store->pin_host_source(request.state);
                state_image = state.data;
            }
        }
    } catch (const std::exception&) { state_image = nullptr; }
    if (state_image == nullptr) {
        release_snapshot(snapshot);
        return std::nullopt;
    }
    snapshot.state       = request.state;
    snapshot.state_image = state_image;
    snapshot.metadata            = std::move(request.metadata.blob);
    const std::size_t blob_bytes = sizeof(PromptCacheRecordHeader) +
                                   static_cast<std::size_t>(state_image_bytes_) +
                                   snapshot.metadata.size();
    if (blob_bytes > std::numeric_limits<std::uint32_t>::max()) {
        release_snapshot(snapshot);
        return std::nullopt;
    }
    snapshot.state_bytes = static_cast<std::uint32_t>(blob_bytes);
    snapshot.header      = PromptCacheRecordHeader{
             .magic                   = kPromptCacheRecordMagic,
             .version                 = kPromptCacheRecordVersion,
             .checkpoint_kind         = static_cast<std::uint32_t>(request.checkpoint.kind),
             .checkpoint_ordinal      = request.checkpoint.ordinal,
             .frontier                = request.frontier,
             .backend_frontier        = request.backend_frontier,
             .main_pages              = main_pages,
             .backend_pages           = backend_pages,
             .main_page_bytes         = page_bytes_,
             .backend_page_bytes      = backend_page_bytes_,
             .state_image_bytes       = state_image_bytes_,
             .execution_frontier      = request.execution_frontier,
             .text_kv_valid           = request.text_kv_valid,
             .mtp_kv_valid            = request.mtp_kv_valid,
             .dflash_context_frontier = request.dflash_context_frontier,
             .rope_delta              = request.rope_delta,
             .ledger_bytes            = request.metadata.ledger_bytes,
             .identity_bytes          = request.metadata.identity_bytes,
             .digest_bytes            = request.metadata.digest_bytes,
             .vision_items            = request.metadata.vision_items,
             .vision_patches          = request.metadata.vision_patches,
    };

    const std::uint64_t token = next_token_++;
    if (next_token_ == 0) { ++next_token_; }
    snapshots_.emplace(token, std::move(snapshot));
    return token;
}

std::optional<std::uint64_t>
PromptCacheTransferPort::prepare_capture(const SequenceState& sequence,
                                         runtime::CheckpointRef checkpoint) {
    if (!sequence.kv) { return std::nullopt; }
    const StateImageHandle* state = nullptr;
    switch (checkpoint.kind) {
    case runtime::CheckpointKind::SessionEndpoint:
        if (!sequence.endpoint_valid || sequence.execution_frontier != checkpoint.frontier) {
            return std::nullopt;
        }
        state = &sequence.state.read;
        break;
    case runtime::CheckpointKind::TurnClosure:
    case runtime::CheckpointKind::ResponseReplay:
        if (!sequence.rewrite_checkpoint.valid || !sequence.rewrite_state ||
            sequence.rewrite_checkpoint.frontier != checkpoint.frontier) {
            return std::nullopt;
        }
        state = &*sequence.rewrite_state;
        break;
    case runtime::CheckpointKind::LongAnchor: {
        const auto anchor = std::find_if(sequence.long_anchors.begin(), sequence.long_anchors.end(),
                                         [&](const LongAnchorCheckpoint& candidate) {
                                             return candidate.ordinal == checkpoint.ordinal &&
                                                    candidate.frontier == checkpoint.frontier;
                                         });
        if (anchor == sequence.long_anchors.end()) { return std::nullopt; }
        state = &anchor->state;
        break;
    }
    case runtime::CheckpointKind::SharedStablePrefix:
        return std::nullopt;
    }
    if (state == nullptr) { return std::nullopt; }
    return build_snapshot(SnapshotRequest{
        .text               = sequence.kv->text,
        .backend            = sequence.kv->backend ? &*sequence.kv->backend : nullptr,
        .state              = *state,
        .checkpoint         = checkpoint,
        .metadata           = build_capture_metadata(sequence.ledger, &sequence.prefix_identity,
                                                     &sequence.prefix_digests, checkpoint.frontier),
        .frontier           = checkpoint.frontier,
        .backend_frontier   = program_.checkpoint_backend_frontier(checkpoint.frontier),
        .execution_frontier = sequence.execution_frontier,
        .text_kv_valid      = sequence.text_kv_valid,
        .mtp_kv_valid       = sequence.mtp_kv_valid,
        .dflash_context_frontier = sequence.dflash_context_frontier,
        .rope_delta              = sequence.rope_delta,
    });
}

std::optional<std::uint64_t>
PromptCacheTransferPort::prepare_capture(const SharedPrefixState& shared,
                                         runtime::CheckpointRef checkpoint) {
    if (!shared.kv || checkpoint.kind != runtime::CheckpointKind::SharedStablePrefix ||
        shared.frontier != checkpoint.frontier) {
        return std::nullopt;
    }
    return build_snapshot(SnapshotRequest{
        .text       = shared.kv->text,
        .backend    = shared.kv->backend ? &*shared.kv->backend : nullptr,
        .state      = shared.state,
        .checkpoint = checkpoint,
        // A shared prefix keeps its exact identity but no digest series, so its record carries
        // the ledger and identity and is not adoptable; shared adoption needs its own slot kind.
        .metadata           = shared.identity ? build_capture_metadata(shared.identity->ledger(),
                                                                       shared.identity->prefix_identity(),
                                                                       nullptr, shared.frontier)
                                              : CaptureMetadata{},
        .frontier           = shared.frontier,
        .backend_frontier   = shared.backend_frontier,
        .execution_frontier = shared.frontier,
        .text_kv_valid      = shared.frontier,
        .mtp_kv_valid       = shared.backend_frontier,
        .dflash_context_frontier = 0,
        .rope_delta              = shared.rope_delta,
    });
}

void PromptCacheTransferPort::release_snapshot(PromptCacheCaptureSnapshot& snapshot) noexcept {
    try {
        for (const LogicalKVPageHandle page : snapshot.pinned_text_pages) {
            program_.text_kv_pages->unpin_source(page);
        }
        if (program_.backend_kv_pages) {
            for (const LogicalKVPageHandle page : snapshot.pinned_backend_pages) {
                program_.backend_kv_pages->unpin_source(page);
            }
        }
    } catch (...) {
        // unpin_source only throws on a stale handle, which cannot happen while the pin itself
        // keeps the page alive. Losing a pin is not recoverable, so fail loudly.
        std::terminate();
    }
    snapshot.pinned_text_pages.clear();
    snapshot.pinned_backend_pages.clear();
    if (snapshot.state_image != nullptr) {
        program_.state_store->unpin_host_source(snapshot.state);
        snapshot.state_image = nullptr;
    }
    snapshot.sources.clear();
    snapshot.metadata.clear();
    snapshot.metadata.shrink_to_fit();
}

void PromptCacheTransferPort::release_finished() noexcept {
    const std::lock_guard<std::mutex> guard(mutex_);
    for (auto entry = snapshots_.begin(); entry != snapshots_.end();) {
        if (entry->second.finished && !entry->second.capturing) {
            release_snapshot(entry->second);
            entry = snapshots_.erase(entry);
        } else {
            ++entry;
        }
    }
}

std::optional<runtime::DiskCheckpointGeometry>
PromptCacheTransferPort::begin_capture(std::uint64_t owner_token) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (active_ != nullptr) { return std::nullopt; }
    const auto entry = snapshots_.find(owner_token);
    if (entry == snapshots_.end() || entry->second.finished) { return std::nullopt; }
    entry->second.capturing = true;
    active_                 = &entry->second;
    return runtime::DiskCheckpointGeometry{
        .page_bytes  = page_bytes_,
        .page_count  = static_cast<std::uint32_t>(active_->sources.size()),
        .state_bytes = active_->state_bytes,
    };
}

std::span<const std::byte> PromptCacheTransferPort::capture_pages(std::uint32_t first_page,
                                                                  std::uint32_t count) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (active_ == nullptr || count == 0 || count > batch_pages_ ||
        static_cast<std::size_t>(first_page) + count > active_->sources.size()) {
        return {};
    }
    std::byte* cursor = capture_staging_.data();
    for (std::uint32_t page = 0; page < count; ++page) {
        const PromptCacheCaptureSnapshot::PageSource& source = active_->sources[first_page + page];
        std::memcpy(cursor, source.data, source.bytes);
        if (source.bytes < page_bytes_) {
            std::memset(cursor + source.bytes, 0, page_bytes_ - source.bytes);
        }
        cursor += page_bytes_;
    }
    return {capture_staging_.data(), static_cast<std::size_t>(count) * page_bytes_};
}

std::span<const std::byte> PromptCacheTransferPort::capture_state_image() {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (active_ == nullptr || active_->state_image == nullptr) { return {}; }
    try {
        state_staging_.assign(active_->state_bytes, std::byte{});
    } catch (const std::bad_alloc&) { return {}; }
    if (state_staging_.size() != active_->state_bytes) { return {}; }
    std::byte* cursor = state_staging_.data();
    std::memcpy(cursor, &active_->header, sizeof(PromptCacheRecordHeader));
    cursor += sizeof(PromptCacheRecordHeader);
    std::memcpy(cursor, active_->state_image, state_image_bytes_);
    cursor += state_image_bytes_;
    if (!active_->metadata.empty()) {
        std::memcpy(cursor, active_->metadata.data(), active_->metadata.size());
    }
    return {state_staging_.data(), state_staging_.size()};
}

void PromptCacheTransferPort::end_capture(bool published) noexcept {
    (void)published;
    const std::lock_guard<std::mutex> guard(mutex_);
    if (active_ == nullptr) { return; }
    active_->capturing = false;
    active_->finished  = true;
    active_            = nullptr;
}

void PromptCacheTransferPort::drop_capture(std::uint64_t owner_token) noexcept {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto entry = snapshots_.find(owner_token);
    if (entry == snapshots_.end()) { return; }
    // A drop can race with the capture the tier is already running for the same token; marking it
    // finished lets the Engine worker release it once `end_capture` has run.
    entry->second.finished = true;
}

std::optional<runtime::DiskCheckpointGeometry>
PromptCacheTransferPort::begin_restore(const runtime::DiskRecordDescriptor& record) {
    const std::lock_guard<std::mutex> guard(mutex_);
    // A completed restore nothing adopted still owns a Host State slot and a Host KV allocation.
    // Starting the next restore is the point at which nothing can consume it any more.
    if (restore_ && restore_->complete) { release_restore(); }
    if (restore_ || record.page_bytes != page_bytes_ || record.page_count == 0 ||
        record.state_bytes < sizeof(PromptCacheRecordHeader) + state_image_bytes_) {
        return std::nullopt;
    }
    const HostKVPageLayout& layout = program_.host_kv_extents->page_layout(*program_.text_kv_pages);
    if (layout.page_stride != page_bytes_) { return std::nullopt; }
    std::optional<HostKVAllocation> pages =
        program_.host_kv_arena->allocate(layout, record.page_count);
    if (!pages) { return std::nullopt; }
    std::optional<qwen3_5::HostStateSlotHandle> slot = program_.host_state_images->allocate();
    if (!slot) { return std::nullopt; }

    Restore restore;
    restore.pages       = std::move(pages);
    restore.view        = program_.host_kv_arena->writable_view(*restore.pages);
    restore.state_slot  = *slot;
    restore.page_count  = record.page_count;
    restore.state_bytes = record.state_bytes;
    restore_            = std::move(restore);
    restore_state_staging_.assign(record.state_bytes, std::byte{});
    return runtime::DiskCheckpointGeometry{
        .page_bytes  = page_bytes_,
        .page_count  = record.page_count,
        .state_bytes = record.state_bytes,
    };
}

std::span<std::byte> PromptCacheTransferPort::restore_staging(std::uint32_t count) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (!restore_) { return {}; }
    // The state image comes first and needs its own buffer: it is longer than one page and is
    // parsed before any of it is placed. Page batches are read straight into the pinned Host KV
    // allocation they will stay in, so a page is never copied twice.
    if (!restore_->state_committed) { return restore_state_staging_; }
    if (count == 0 || count > batch_pages_ ||
        static_cast<std::size_t>(restore_->page_cursor) + count > restore_->page_count) {
        return {};
    }
    return {restore_->view.data() + static_cast<std::size_t>(restore_->page_cursor) * page_bytes_,
            static_cast<std::size_t>(count) * page_bytes_};
}

bool PromptCacheTransferPort::commit_pages(std::uint32_t first_page, std::uint32_t count) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (!restore_ || !restore_->state_committed || first_page != restore_->page_cursor ||
        count == 0 || static_cast<std::size_t>(first_page) + count > restore_->page_count) {
        return false;
    }
    restore_->page_cursor += count;
    return true;
}

bool PromptCacheTransferPort::commit_state_image(std::span<const std::byte> payload) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (!restore_ || restore_->state_committed || payload.size() != restore_->state_bytes) {
        return false;
    }
    PromptCacheRecordHeader header{};
    std::memcpy(&header, payload.data(), sizeof(header));
    // The store header signature already refuses a foreign model. This second check catches a
    // record written by the same signature but a different record layout version, which is the
    // one way a matching store can still hold bytes this build cannot place. A version 1 record
    // fails here, which is the whole point of the version field: its blob is shorter than this
    // header, so nothing after `magic` would mean what it says.
    if (header.magic != kPromptCacheRecordMagic || header.version != kPromptCacheRecordVersion ||
        header.main_page_bytes != page_bytes_ || header.backend_page_bytes != backend_page_bytes_ ||
        header.state_image_bytes != state_image_bytes_ || header.frontier == 0) {
        return false;
    }
    const std::size_t expected =
        static_cast<std::size_t>(header.main_pages) +
        static_cast<std::size_t>(header.backend_pages) * backend_record_pages_;
    if (expected != restore_->page_count) { return false; }
    const std::size_t metadata_bytes =
        static_cast<std::size_t>(header.ledger_bytes) + header.identity_bytes + header.digest_bytes;
    if (sizeof(header) + state_image_bytes_ + metadata_bytes != payload.size()) { return false; }
    std::vector<TokenId> ledger;
    qwen3_5::detail::ResidentPrefixIdentity identity;
    qwen3_5::detail::PrefixShortlistDigests digests;
    if (metadata_bytes != 0 &&
        !decode_metadata(header, payload.subspan(sizeof(header) + state_image_bytes_), ledger,
                         identity, digests)) {
        return false;
    }
    const qwen3_5::HostStateImageView destination =
        program_.host_state_images->writable_view(*restore_->state_slot);
    if (destination.data == nullptr) { return false; }
    std::memcpy(destination.data, payload.data() + sizeof(header), state_image_bytes_);
    restore_->header          = header;
    restore_->ledger          = std::move(ledger);
    restore_->identity.swap(identity);
    restore_->digests.swap(digests);
    restore_->state_committed = true;
    return true;
}

bool PromptCacheTransferPort::decode_metadata(const PromptCacheRecordHeader& header,
                                              std::span<const std::byte> payload,
                                              std::vector<TokenId>& ledger,
                                              qwen3_5::detail::ResidentPrefixIdentity& identity,
                                              qwen3_5::detail::PrefixShortlistDigests& digests) {
    if (header.ledger_bytes % sizeof(TokenId) != 0 ||
        header.ledger_bytes / sizeof(TokenId) != header.frontier) {
        return false;
    }
    ledger.resize(header.frontier);
    if (header.frontier != 0) { std::memcpy(ledger.data(), payload.data(), header.ledger_bytes); }
    if (!identity.decode(payload.subspan(header.ledger_bytes, header.identity_bytes)) ||
        identity.size() != header.frontier) {
        return false;
    }
    // A shared-prefix record carries no digest series. It is decoded as empty and the adoption
    // transaction refuses it there, where the reason can be named.
    return header.digest_bytes == 0 ||
           (digests.decode(payload.subspan(static_cast<std::size_t>(header.ledger_bytes) +
                                               header.identity_bytes,
                                           header.digest_bytes)) &&
            digests.size() == header.frontier);
}

void PromptCacheTransferPort::end_restore(bool complete) noexcept {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (!restore_) { return; }
    // A complete restore is kept for the adoption transaction, which is the only thing that can
    // turn it into a continuation; anything short of complete has no consumer.
    if (!complete || !restore_->state_committed || restore_->page_cursor != restore_->page_count) {
        release_restore();
        return;
    }
    restore_->complete = true;
    restore_->view     = {};
    restore_state_staging_.clear();
    restore_state_staging_.shrink_to_fit();
}

std::optional<PromptCacheRestoredRecord> PromptCacheTransferPort::take_restored_record() noexcept {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (!restore_ || !restore_->complete || !restore_->pages || !restore_->state_slot) {
        return std::nullopt;
    }
    PromptCacheRestoredRecord out;
    out.pages      = std::move(*restore_->pages);
    out.state_slot = *restore_->state_slot;
    out.header     = restore_->header;
    out.ledger     = std::move(restore_->ledger);
    out.identity.swap(restore_->identity);
    out.digests.swap(restore_->digests);
    restore_->pages.reset();
    restore_->state_slot.reset();
    restore_.reset();
    return out;
}

void PromptCacheTransferPort::discard_restored_record() noexcept {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (restore_ && restore_->complete) { release_restore(); }
}

void PromptCacheTransferPort::release_restore() noexcept {
    if (!restore_) { return; }
    if (restore_->state_slot) { (void)program_.host_state_images->release(*restore_->state_slot); }
    restore_->view = {};
    if (restore_->pages) { (void)restore_->pages->release(); }
    restore_.reset();
    restore_state_staging_.clear();
    restore_state_staging_.shrink_to_fit();
}

} // namespace ninfer::models::qwen3_5::detail
