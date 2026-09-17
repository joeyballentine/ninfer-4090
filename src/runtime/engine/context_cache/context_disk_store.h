#pragma once

// Persistent third tier of the context cache: Device checkpoint -> pinned Host State slot ->
// Disk record.
//
// This file owns the on-disk representation and its policy (publication, LRU capacity,
// mark-and-sweep compaction). It is deliberately payload-agnostic: a record is a prefix identity
// plus one State image and an ordered list of fixed-size KV page images, all opaque bytes. The
// Program keeps ownership of what those bytes mean.
//
// Layout of a cache directory:
//
//   index.log   append-only journal: a sector-sized header, then length-and-checksum framed
//               entries (BlobCommit, RecordPublish, RecordDrop, RecordTouch).
//   blobs.dat   append-only, sector-aligned, content-addressed extents shared by every record.
//               Two turns of one conversation, or two branches of it, reference the same page
//               extents; that sharing is what makes a per-turn spill cheap.
//
// A segment file plus one journal was chosen over a file per record because a 77k-token record
// is ~1200 KV pages: a file per page loses to one sequential extent stream, and a file per
// record loses the cross-turn sharing that makes the common case a small delta write.
//
// Durability rule: a record exists only once its RecordPublish entry is complete, checksummed
// and flushed, and that entry is appended only after the blob bytes it names are themselves
// flushed. Replay stops at the first incomplete or mis-checksummed entry and truncates there,
// so a torn tail can never publish a record whose payload is missing.

#include "core/positional_file.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime {

// Mirrors the model's `PrefixShortlistKey`: two rolling content digests over the token frontier,
// the frontier itself and the identity tag that separates incompatible token/position identities.
// The runtime converts the model key into this one so the disk tier keeps no model dependency.
struct DiskRecordKey {
    std::uint64_t digest_low  = 0;
    std::uint64_t digest_high = 0;
    std::uint32_t frontier    = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(DiskRecordKey, DiskRecordKey) noexcept = default;
};

struct DiskRecordKeyHash {
    [[nodiscard]] std::size_t operator()(const DiskRecordKey& key) const noexcept;
};

// 128-bit content address of one blob extent.
struct DiskBlobHash {
    std::uint64_t low  = 0;
    std::uint64_t high = 0;

    [[nodiscard]] friend constexpr bool operator==(DiskBlobHash, DiskBlobHash) noexcept = default;
};

struct DiskBlobHashHasher {
    [[nodiscard]] std::size_t operator()(const DiskBlobHash& hash) const noexcept {
        return static_cast<std::size_t>(hash.low ^ (hash.high * 0x9e3779b97f4a7c15ULL));
    }
};

[[nodiscard]] DiskBlobHash disk_blob_hash(std::span<const std::byte> payload) noexcept;

// What a lookup returns. It is a stable snapshot: the extents it names stay live until the
// holder releases it, because eviction and compaction only retire unpinned records.
struct DiskRecordDescriptor {
    DiskRecordKey key;
    std::uint32_t page_bytes  = 0;
    std::uint32_t page_count  = 0;
    std::uint32_t state_bytes = 0;
    std::uint64_t payload_bytes = 0;
    std::uint64_t sequence      = 0;
};

// Where one payload lives in the extent file. The portable path never needs this - it reads
// through the store - but a kernel-bypass DMA engine is handed file offsets directly.
struct DiskExtentLocation {
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

struct ContextDiskStoreStats {
    std::uint64_t records           = 0;
    std::uint64_t blobs             = 0;
    std::uint64_t live_bytes        = 0;
    std::uint64_t dead_bytes        = 0;
    std::uint64_t dead_blobs        = 0;
    std::uint64_t file_bytes        = 0;
    std::uint64_t lookups           = 0;
    std::uint64_t hits              = 0;
    std::uint64_t records_published = 0;
    std::uint64_t records_evicted   = 0;
    std::uint64_t records_rejected  = 0;
    std::uint64_t compactions       = 0;
    std::uint64_t bytes_written     = 0;
    std::uint64_t bytes_read        = 0;
};

struct ContextDiskStoreConfig {
    std::filesystem::path directory;
    // Artifact plus KV-layout signature. A store whose header signature differs is foreign and
    // its records are never matched, so a different model or KV storage cannot be restored into.
    std::string signature;
    std::uint64_t max_bytes = 30ULL << 30;
    // Capacity eviction runs in one pass from the cap down to this fraction, instead of evicting
    // one record per publish and thrashing the oldest conversation out turn by turn.
    std::uint32_t eviction_low_watermark_percent = 75;
    // Compaction guards. All three must hold; rewriting a multi-gigabyte extent file to reclaim
    // a handful of dead pages costs more device wear than the space is worth.
    std::uint64_t compaction_min_dead_blobs = 256;
    std::uint64_t compaction_min_dead_bytes = 2ULL << 30;
    std::uint32_t compaction_min_dead_percent = 25;
};

// Cooperative abort for the background writer and the sweep. It is a latency-priority signal
// raised when a request arrives, never an invalidation: a spill whose bytes are already written
// still publishes, because the abort at that point would save no I/O and would orphan the
// extents it just appended.
class DiskCancellationToken {
public:
    void request() noexcept { requested_.store(true, std::memory_order_relaxed); }
    void clear() noexcept { requested_.store(false, std::memory_order_relaxed); }
    [[nodiscard]] bool requested() const noexcept {
        return requested_.load(std::memory_order_relaxed);
    }

private:
    std::atomic<bool> requested_{false};
};

class ContextDiskStore {
public:
    explicit ContextDiskStore(ContextDiskStoreConfig config);
    ~ContextDiskStore();
    ContextDiskStore(const ContextDiskStore&)            = delete;
    ContextDiskStore& operator=(const ContextDiskStore&) = delete;

    [[nodiscard]] const ContextDiskStoreConfig& config() const noexcept { return config_; }

    // Streams one record. The caller supplies pages in whatever batch size its staging buffer
    // allows; the writer never holds more than the current page.
    class RecordWriter {
    public:
        RecordWriter(RecordWriter&&) noexcept;
        ~RecordWriter();
        RecordWriter(const RecordWriter&)            = delete;
        RecordWriter& operator=(const RecordWriter&) = delete;
        RecordWriter& operator=(RecordWriter&&)      = delete;

        void write_state_image(std::span<const std::byte> payload);
        void write_page(std::span<const std::byte> payload);
        // Flushes the extents, then appends and flushes the publication entry. False means the
        // record was not published and its extents remain unreferenced.
        [[nodiscard]] bool publish();
        void abandon() noexcept;

    private:
        friend class ContextDiskStore;
        RecordWriter(ContextDiskStore& store, DiskRecordKey key, std::uint32_t page_bytes);

        ContextDiskStore* store_ = nullptr;
        DiskRecordKey key_;
        std::uint32_t page_bytes_  = 0;
        std::uint32_t state_bytes_ = 0;
        bool has_state_            = false;
        bool finished_             = false;
        std::vector<DiskBlobHash> page_hashes_;
        DiskBlobHash state_hash_;
        std::vector<DiskBlobHash> appended_;
    };

    [[nodiscard]] RecordWriter begin_record(DiskRecordKey key, std::uint32_t page_bytes);

    // Non-blocking against the executor: takes the index mutex only, never file I/O.
    [[nodiscard]] std::optional<DiskRecordDescriptor> lookup(const DiskRecordKey& key);
    // Longest published prefix among the supplied candidate keys, which the caller derives from
    // the incoming prompt at each reusable frontier.
    [[nodiscard]] std::optional<DiskRecordDescriptor>
    lookup_longest(std::span<const DiskRecordKey> candidates);

    [[nodiscard]] const std::filesystem::path& extent_file() const noexcept { return blobs_path_; }
    // State image first when the record has one, then the pages in order.
    [[nodiscard]] std::vector<DiskExtentLocation>
    extent_locations(const DiskRecordDescriptor& record) const;

    void read_state_image(const DiskRecordDescriptor& record, std::span<std::byte> destination);
    void read_page(const DiskRecordDescriptor& record, std::uint32_t page_index,
                   std::span<std::byte> destination);

    void drop(const DiskRecordKey& key);
    // Evicts least-recently-used records until live bytes fall under the low watermark.
    void enforce_capacity();

    [[nodiscard]] bool compaction_due() const;
    // Rewrites both files keeping only referenced extents. File copying runs without the index
    // mutex; the mutex is taken only for the final offset swap, so lookups stay available.
    bool compact(const DiskCancellationToken& cancel);

    [[nodiscard]] ContextDiskStoreStats stats() const;

private:
    struct BlobExtent {
        std::uint64_t offset   = 0;
        std::uint64_t bytes    = 0;
        std::uint64_t aligned  = 0;
        std::uint32_t refcount = 0;
    };

    struct RecordEntry {
        DiskRecordDescriptor descriptor;
        bool has_state = false;
        DiskBlobHash state_hash;
        std::vector<DiskBlobHash> page_hashes;
        std::uint64_t access_sequence = 0;
    };

    void open_store();
    void reset_store();
    void replay_index(std::uint64_t payload_offset);
    void append_entry(std::uint16_t kind, std::span<const std::byte> payload);
    [[nodiscard]] DiskBlobHash append_blob(std::span<const std::byte> payload,
                                           std::vector<DiskBlobHash>& appended);
    void release_blob(const DiskBlobHash& hash);
    void retain_blob(const DiskBlobHash& hash);
    void erase_record_locked(const DiskRecordKey& key, bool journal);
    [[nodiscard]] bool publish_record(RecordWriter& writer);
    void read_extent(const DiskBlobHash& hash, std::span<std::byte> destination);
    void rewrite_index(const std::filesystem::path& path,
                       const std::vector<std::pair<DiskBlobHash, BlobExtent>>& blobs);

    ContextDiskStoreConfig config_;
    mutable std::mutex mutex_;
    core::PositionalFile index_;
    core::PositionalFile blobs_;
    std::filesystem::path index_path_;
    std::filesystem::path blobs_path_;
    std::unordered_map<DiskBlobHash, BlobExtent, DiskBlobHashHasher> blob_index_;
    std::unordered_map<DiskRecordKey, RecordEntry, DiskRecordKeyHash> records_;
    std::uint64_t access_clock_  = 0;
    std::uint64_t publish_clock_ = 0;
    std::uint64_t live_bytes_    = 0;
    std::uint64_t dead_bytes_    = 0;
    std::uint64_t dead_blobs_    = 0;
    core::AlignedBuffer staging_;
    ContextDiskStoreStats stats_;
};

} // namespace ninfer::runtime
