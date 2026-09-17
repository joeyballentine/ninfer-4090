#pragma once

// Types shared between the persistent context tier and the Program that performs its transfers.
//
// The tier and its store live in the Engine (`runtime/engine/context_cache/`), but every physical
// byte movement belongs to the Program, which owns the pinned Host replicas a record is written
// from and read back into. This header is the only thing the two sides agree on, so the model
// layer never includes an Engine header and the Engine never learns what a page contains.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>

namespace ninfer::runtime {

// Mirrors the model's `PrefixShortlistKey`: two rolling content digests over the token frontier,
// the frontier itself and the identity tag that separates incompatible token/position identities.
// The runtime converts the model key into this one so the disk tier keeps no model dependency.
struct DiskRecordKey {
    std::uint64_t digest_low   = 0;
    std::uint64_t digest_high  = 0;
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(DiskRecordKey, DiskRecordKey) noexcept = default;
};

// What a lookup returns. It is a stable snapshot: the extents it names stay live until the
// holder releases it, because eviction and compaction only retire unpinned records.
struct DiskRecordDescriptor {
    DiskRecordKey key;
    std::uint32_t page_bytes    = 0;
    std::uint32_t page_count    = 0;
    std::uint32_t state_bytes   = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t sequence      = 0;
};

// Where one payload lives in the extent file. The portable path never needs this - it reads
// through the store - but a kernel-bypass DMA engine is handed file offsets directly.
struct DiskExtentLocation {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

// Shape of one checkpoint as the store sees it. `page_bytes` is the Host page-group stride of the
// main KV pool; it is never a constant, because a per-layer KV schedule gives every layer its own
// plane byte size and only the page-group stride covers all of them.
struct DiskCheckpointGeometry {
    std::uint32_t page_bytes  = 0;
    std::uint32_t page_count  = 0;
    std::uint32_t state_bytes = 0;
};

// The physical half of a spill or restore.
//
// Thread contract. `begin_capture` / `capture_pages` / `capture_state_image` / `end_capture` /
// `drop_capture` run on the tier's I/O thread, concurrently with the Engine worker's decode
// round, and must therefore touch no mutable Program state: the implementation resolves
// `owner_token` against a snapshot the Engine worker registered and pinned beforehand. The
// restore methods run on the thread that asked for the restore, which is the Engine worker
// itself, and may use the Program's transfer stream and completion events.
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
    [[nodiscard]] virtual std::span<const std::byte> capture_state_image()              = 0;
    virtual void end_capture(bool published) noexcept                                   = 0;
    // A queued spill that will never reach `begin_capture`: superseded, dropped by a full queue,
    // cancelled by an arriving request or discarded at shutdown. Without it the snapshot the
    // Engine pinned for that token would stay pinned forever.
    virtual void drop_capture(std::uint64_t owner_token) noexcept = 0;

    // Restore. A disengaged result means no destination could be reserved, so the admission path
    // falls back to prefill.
    [[nodiscard]] virtual std::optional<DiskCheckpointGeometry>
    begin_restore(const DiskRecordDescriptor& record) = 0;
    // Staging the tier reads `count` page images into before the port commits them.
    [[nodiscard]] virtual std::span<std::byte> restore_staging(std::uint32_t count)        = 0;
    [[nodiscard]] virtual bool commit_pages(std::uint32_t first_page, std::uint32_t count) = 0;
    [[nodiscard]] virtual bool commit_state_image(std::span<const std::byte> payload)      = 0;
    virtual void end_restore(bool complete) noexcept                                       = 0;

#if defined(_WIN32) && defined(NINFER_DIRECTSTORAGE)
    // Optional kernel-bypass restore. DirectStorage reads the extent file straight into device
    // memory, skipping the pinned staging round trip the portable path takes; on the donor fork
    // that turned a 2.2 GB, 152k-token restore into 367 ms. It is declared here and nowhere
    // implemented: it needs a Windows SDK, a D3D12 device and a real GPU, none of which this
    // tree can build or test against. The portable path above is complete without it.
    //
    // The lesson to carry over when it is implemented (donor 4a0022d6): the D3D12 fence shared
    // with CUDA must be waited on from the CPU thread before any staging resource is released.
    // Releasing COM resources while the DirectStorage queue still has requests in flight
    // crashes inside the NVIDIA D3D12 driver. Tear the CUDA side down with
    // cudaDestroyExternalMemory and never cudaFree a mapped external pointer.
    //
    // Returns false to fall back to the portable path for this record.
    [[nodiscard]] virtual bool
    try_direct_storage_restore(const DiskRecordDescriptor& record,
                               const std::filesystem::path& extent_file,
                               std::span<const DiskExtentLocation> extents) = 0;
#endif
};

} // namespace ninfer::runtime
