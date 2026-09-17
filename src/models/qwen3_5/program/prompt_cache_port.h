#pragma once

// Physical half of the persistent prompt cache: the Program side of `ContextDiskTransferPort`.
//
// The Engine owns when a record is written or read; every byte it moves is moved here, because
// the pinned Host State slot and the pinned Host KV extents a record is made of belong to the
// Program. The port therefore lives with the rest of the mutable context state instead of in the
// Engine, and the Engine only adapts it to the tier.
//
// Threading. `ProgramImpl` is the single mutation owner and is not thread safe, while the tier
// writes on its own I/O thread. The capture side is split accordingly:
//
//   * `prepare_capture` runs on the Engine worker. It verifies that both the State image and
//     every KV page of one checkpoint already have a Host replica, pins those replicas, records
//     their pinned addresses and returns a fresh capture token.
//   * `begin_capture` / `capture_pages` / `capture_state_image` / `end_capture` / `drop_capture`
//     run on the I/O thread. They touch only this object's own mutex, the recorded addresses and
//     a heap staging buffer - no `ProgramImpl` state, no CUDA call, no transfer stream. That is
//     why a spill never contends with the decode round.
//   * `release_finished` runs on the Engine worker and unpins what the I/O thread has finished.
//
// A Device-only checkpoint is never captured: reading it back would need the transfer stream the
// decode round is using. That restriction is what lets the whole capture side avoid CUDA.
//
// The restore side runs on the caller's thread, which is the Engine worker, so it may use the
// Program's stores directly.
//
// Record shape. One record page is one main-pool page group: `page_bytes` is that pool's Host
// page stride, never a constant, because a per-layer KV schedule gives every layer its own plane
// byte size and only the page-group stride covers all of them. A backend-pool page has its own
// stride, so it occupies `ceil(backend_stride / page_bytes)` consecutive record pages with the
// tail zero filled. Main pages come first, then backend pages, which keeps the main pages of two
// turns of one conversation byte-identical and therefore shared by the store's content
// addressing.

#include "runtime/contract/context_disk.h"
#include "runtime/contract/resources.h"
#include "models/qwen3_5/program/storage/host_kv_store.h"
#include "models/qwen3_5/program/storage/state_store.h"
#include "models/qwen3_5/state/state_image.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

class ProgramImpl;
struct SequenceState;
struct SharedPrefixState;

// Fixed head of a record's state-image blob. It makes a record self describing, so a store whose
// header signature happens to match but whose geometry does not is rejected on read instead of
// producing a silently wrong restore.
struct PromptCacheRecordHeader {
    std::uint32_t magic                   = 0;
    std::uint32_t version                 = 0;
    std::uint32_t checkpoint_kind         = 0;
    std::uint32_t checkpoint_ordinal      = 0;
    std::uint32_t frontier                = 0;
    std::uint32_t backend_frontier        = 0;
    std::uint32_t main_pages              = 0;
    std::uint32_t backend_pages           = 0;
    std::uint32_t main_page_bytes         = 0;
    std::uint32_t backend_page_bytes      = 0;
    std::uint32_t state_image_bytes       = 0;
    std::uint32_t execution_frontier      = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::int32_t rope_delta               = 0;
};

inline constexpr std::uint32_t kPromptCacheRecordMagic   = 0x3552'4351U; // "QCR5"
inline constexpr std::uint32_t kPromptCacheRecordVersion = 1;

// Host bytes of one captured checkpoint, resolved and pinned while the Engine worker still owns
// the Program. `state_image` and every `sources` entry point into pinned Host memory that the
// pins below keep alive and in place until the capture is released.
struct PromptCacheCaptureSnapshot {
    struct PageSource {
        const std::byte* data = nullptr;
        // Payload of this record page. A backend page whose stride is not a multiple of the
        // record page size leaves the remainder zero filled.
        std::uint32_t bytes = 0;
    };

    std::vector<PageSource> sources;
    std::vector<LogicalKVPageHandle> pinned_text_pages;
    std::vector<LogicalKVPageHandle> pinned_backend_pages;
    StateImageHandle state;
    const std::byte* state_image = nullptr;
    PromptCacheRecordHeader header;
    std::uint32_t state_bytes = 0;
    bool capturing            = false;
    bool finished             = false;
};

class PromptCacheTransferPort final : public runtime::ContextDiskTransferPort {
public:
    explicit PromptCacheTransferPort(ProgramImpl& program);
    ~PromptCacheTransferPort() override;

    // --- Engine worker -----------------------------------------------------------------------

    // Pins one checkpoint's Host replicas and returns the token the spill request must carry.
    // Disengaged when the checkpoint is not fully Host resident, when its geometry is not
    // capturable, or when too many captures are already outstanding.
    [[nodiscard]] std::optional<std::uint64_t> prepare_capture(const SequenceState& sequence,
                                                               runtime::CheckpointRef checkpoint);
    [[nodiscard]] std::optional<std::uint64_t> prepare_capture(const SharedPrefixState& shared,
                                                               runtime::CheckpointRef checkpoint);
    // Unpins every capture the I/O thread has finished or dropped. Idempotent.
    void release_finished() noexcept;
    [[nodiscard]] std::size_t outstanding_captures() const noexcept;

    // --- ContextDiskTransferPort -------------------------------------------------------------

    [[nodiscard]] std::uint32_t batch_pages() const noexcept override { return batch_pages_; }

    [[nodiscard]] std::optional<runtime::DiskCheckpointGeometry>
    begin_capture(std::uint64_t owner_token) override;
    [[nodiscard]] std::span<const std::byte> capture_pages(std::uint32_t first_page,
                                                           std::uint32_t count) override;
    [[nodiscard]] std::span<const std::byte> capture_state_image() override;
    void end_capture(bool published) noexcept override;
    void drop_capture(std::uint64_t owner_token) noexcept override;

    [[nodiscard]] std::optional<runtime::DiskCheckpointGeometry>
    begin_restore(const runtime::DiskRecordDescriptor& record) override;
    [[nodiscard]] std::span<std::byte> restore_staging(std::uint32_t count) override;
    [[nodiscard]] bool commit_pages(std::uint32_t first_page, std::uint32_t count) override;
    [[nodiscard]] bool commit_state_image(std::span<const std::byte> payload) override;
    void end_restore(bool complete) noexcept override;

private:
    struct Restore {
        std::optional<HostKVAllocation> pages;
        std::optional<qwen3_5::HostStateSlotHandle> state_slot;
        HostKVAllocationView view;
        PromptCacheRecordHeader header;
        std::uint32_t page_count  = 0;
        std::uint32_t page_cursor = 0;
        std::uint32_t state_bytes = 0;
        bool state_committed      = false;
    };

    struct SnapshotRequest {
        KVAddressSpaceHandle text;
        const KVAddressSpaceHandle* backend = nullptr;
        StateImageHandle state;
        runtime::CheckpointRef checkpoint;
        std::uint32_t frontier                = 0;
        std::uint32_t backend_frontier        = 0;
        std::uint32_t execution_frontier      = 0;
        std::uint32_t text_kv_valid           = 0;
        std::uint32_t mtp_kv_valid            = 0;
        std::uint32_t dflash_context_frontier = 0;
        std::int32_t rope_delta               = 0;
    };

    [[nodiscard]] std::optional<std::uint64_t> build_snapshot(const SnapshotRequest& request);
    // Appends the Host addresses of `frontier`'s pages, pinning each one. Returns false and pins
    // nothing further when any page is missing a usable Host replica.
    [[nodiscard]] bool collect_pages(KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                                     KVAddressSpaceHandle address, std::uint32_t frontier,
                                     std::uint32_t stride, std::vector<LogicalKVPageHandle>& pinned,
                                     std::vector<PromptCacheCaptureSnapshot::PageSource>& sources);
    void release_snapshot(PromptCacheCaptureSnapshot& snapshot) noexcept;
    void release_restore() noexcept;

    ProgramImpl& program_;
    std::uint32_t page_bytes_           = 0;
    std::uint32_t backend_page_bytes_   = 0;
    std::uint32_t backend_record_pages_ = 0;
    std::uint32_t batch_pages_          = 1;
    std::uint32_t state_image_bytes_    = 0;

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, PromptCacheCaptureSnapshot> snapshots_;
    std::uint64_t next_token_           = 1;
    PromptCacheCaptureSnapshot* active_ = nullptr;
    std::vector<std::byte> capture_staging_;
    std::vector<std::byte> state_staging_;

    std::optional<Restore> restore_;
    std::vector<std::byte> restore_state_staging_;
};

} // namespace ninfer::models::qwen3_5::detail
