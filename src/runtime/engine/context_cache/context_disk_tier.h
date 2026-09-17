#pragma once

// Policy for the third context tier. Device checkpoint -> pinned Host State slot -> Disk record.
//
// The tier owns when a record is written, when one is read back and how much work either costs.
// It owns no device memory and no knowledge of what a page contains: every physical transfer
// goes through `ContextDiskTransferPort`, which the Engine implements over the Program's pinned
// staging and transfer stream. That split keeps Program as the only physical authority and lets
// the whole state machine be exercised on a host with no GPU.
//
// Transfers are batched. A 77k-token record is on the order of a thousand KV pages; staging the
// whole record at once is what made the donor fork fail on a 24 GB card (c15e0e9e), so the tier
// asks the port for `batch_pages()` pages at a time and never holds more.

#include "runtime/engine/context_cache/context_disk_store.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <span>
#include <thread>

namespace ninfer::runtime {

struct DiskCheckpointGeometry {
    std::uint32_t page_bytes  = 0;
    std::uint32_t page_count  = 0;
    std::uint32_t state_bytes = 0;
};

// The physical half of a spill or restore. Every method runs on the tier's I/O thread except
// the restore methods, which run on the thread that asked for the restore.
class ContextDiskTransferPort {
public:
    virtual ~ContextDiskTransferPort() = default;

    // Upper bound on pages staged at once. The tier never asks for more in one call.
    [[nodiscard]] virtual std::uint32_t batch_pages() const noexcept = 0;

    // Spill. `owner_token` names the host state slot the Engine is evicting; the port resolves it
    // against the Program. A disengaged result means the slot is gone and the spill is dropped.
    [[nodiscard]] virtual std::optional<DiskCheckpointGeometry>
    begin_capture(std::uint64_t owner_token) = 0;
    // Returns staging holding `count` consecutive page images, or an empty span on failure.
    [[nodiscard]] virtual std::span<const std::byte> capture_pages(std::uint32_t first_page,
                                                                   std::uint32_t count) = 0;
    [[nodiscard]] virtual std::span<const std::byte> capture_state_image() = 0;
    virtual void end_capture(bool published) noexcept = 0;

    // Restore. A disengaged result means no destination could be reserved, so the admission path
    // falls back to prefill.
    [[nodiscard]] virtual std::optional<DiskCheckpointGeometry>
    begin_restore(const DiskRecordDescriptor& record) = 0;
    // Staging the tier reads `count` page images into before the port commits them.
    [[nodiscard]] virtual std::span<std::byte> restore_staging(std::uint32_t count) = 0;
    [[nodiscard]] virtual bool commit_pages(std::uint32_t first_page, std::uint32_t count) = 0;
    [[nodiscard]] virtual bool commit_state_image(std::span<const std::byte> payload) = 0;
    virtual void end_restore(bool complete) noexcept = 0;
};

struct DiskSpillRequest {
    DiskRecordKey key;
    // Opaque Engine handle for the host state slot being evicted.
    std::uint64_t owner_token = 0;
};

struct ContextDiskTierStats {
    std::uint64_t spills_requested = 0;
    std::uint64_t spills_published = 0;
    std::uint64_t spills_dropped   = 0;
    std::uint64_t spilled_bytes    = 0;
    std::uint64_t lookups          = 0;
    std::uint64_t hits             = 0;
    std::uint64_t restores         = 0;
    std::uint64_t restore_failures = 0;
    std::uint64_t restored_bytes   = 0;
    ContextDiskStoreStats store;
};

class ContextDiskTier {
public:
    ContextDiskTier(ContextDiskStoreConfig config, ContextDiskTransferPort& port,
                    std::uint32_t queue_depth = 4);
    ~ContextDiskTier();
    ContextDiskTier(const ContextDiskTier&)            = delete;
    ContextDiskTier& operator=(const ContextDiskTier&) = delete;

    // Hook (a): a host state slot is being evicted. Returns immediately; the decode round is
    // never blocked by disk work. A full queue drops the oldest pending spill, because a newer
    // frontier is strictly more useful than an older one from the same conversation.
    void request_spill(const DiskSpillRequest& request);

    // Hook (b): prefix lookup missed Device and Host. Index only, no file I/O, safe to call from
    // the admission path.
    [[nodiscard]] std::optional<DiskRecordDescriptor>
    lookup(std::span<const DiskRecordKey> candidates);

    // Hook (b) continued: bring the record back before prefill so only the uncovered suffix is
    // computed. Runs on the calling thread and returns false when the port declined, in which
    // case the caller proceeds as if the lookup had missed.
    [[nodiscard]] bool restore(const DiskRecordDescriptor& record);

    // Latency-priority signal raised when a request arrives. It aborts a queued spill that has
    // not started and a sweep that is still copying; it never discards work already written.
    void note_request_arrival() noexcept;
    void clear_request_priority() noexcept { cancel_.clear(); }

    // Blocks until the spill queue is empty. Test and shutdown hook only.
    void drain();

    [[nodiscard]] ContextDiskTierStats stats() const;

private:
    void worker();
    void run_spill(const DiskSpillRequest& request);

    ContextDiskStore store_;
    ContextDiskTransferPort& port_;
    std::uint32_t queue_depth_;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable idle_;
    std::deque<DiskSpillRequest> queue_;
    bool running_ = true;
    bool busy_    = false;
    ContextDiskTierStats stats_;

    // Serializes file access: the store's files are positional but a sweep rewrites them, so
    // only one of {spill batch, restore, sweep} may touch them at a time.
    std::mutex io_mutex_;
    DiskCancellationToken cancel_;
    std::thread worker_;
};

} // namespace ninfer::runtime
