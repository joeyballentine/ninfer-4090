#include "runtime/engine/context_cache/context_disk_store.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {
namespace {

static_assert(std::endian::native == std::endian::little,
              "the context disk store stores fixed-width little-endian fields");

constexpr std::uint64_t kStoreMagic  = 0x314B534449464E4EULL; // "NNIFDSK1"
constexpr std::uint32_t kStoreFormat = 1;
constexpr std::uint64_t kHeaderBytes = core::kDirectIoAlignment;
constexpr std::size_t kMaxSignatureBytes = 1024;

constexpr std::uint16_t kEntryBlobCommit   = 1;
constexpr std::uint16_t kEntryRecordPublish = 2;
constexpr std::uint16_t kEntryRecordDrop    = 3;
constexpr std::uint16_t kEntryRecordTouch   = 4;

constexpr std::size_t kEntryHeaderBytes = 16;

[[nodiscard]] std::uint64_t fnv1a64(std::span<const std::byte> payload,
                                    std::uint64_t seed = 0xcbf29ce484222325ULL) noexcept {
    std::uint64_t hash = seed;
    for (const std::byte byte : payload) {
        hash ^= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(byte));
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

[[nodiscard]] std::uint64_t mix64(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return value;
}

// Bounds-checked reader. A journal entry whose checksum is valid can still be structurally
// inconsistent after a partial rewrite, so every field read is length checked and a short entry
// is reported rather than trusted.
class Cursor {
public:
    explicit Cursor(std::span<const std::byte> data) noexcept : data_(data) {}

    [[nodiscard]] bool truncated() const noexcept { return truncated_; }
    [[nodiscard]] std::size_t position() const noexcept { return position_; }

    template <class T> [[nodiscard]] T read() noexcept {
        T value{};
        if (data_.size() - position_ < sizeof(T)) {
            truncated_ = true;
            return value;
        }
        std::memcpy(&value, data_.data() + position_, sizeof(T));
        position_ += sizeof(T);
        return value;
    }

private:
    std::span<const std::byte> data_;
    std::size_t position_ = 0;
    bool truncated_       = false;
};

void put_u32(std::vector<std::byte>& out, std::uint32_t value) {
    const auto offset = out.size();
    out.resize(offset + sizeof(value));
    std::memcpy(out.data() + offset, &value, sizeof(value));
}

void put_u64(std::vector<std::byte>& out, std::uint64_t value) {
    const auto offset = out.size();
    out.resize(offset + sizeof(value));
    std::memcpy(out.data() + offset, &value, sizeof(value));
}

void put_key(std::vector<std::byte>& out, const DiskRecordKey& key) {
    put_u64(out, key.digest_low);
    put_u64(out, key.digest_high);
    put_u32(out, key.frontier);
    put_u32(out, key.identity_tag);
}

[[nodiscard]] DiskRecordKey take_key(Cursor& cursor) {
    DiskRecordKey key;
    key.digest_low   = cursor.read<std::uint64_t>();
    key.digest_high  = cursor.read<std::uint64_t>();
    key.frontier     = cursor.read<std::uint32_t>();
    key.identity_tag = cursor.read<std::uint32_t>();
    return key;
}

void put_hash(std::vector<std::byte>& out, const DiskBlobHash& hash) {
    put_u64(out, hash.low);
    put_u64(out, hash.high);
}

[[nodiscard]] DiskBlobHash take_hash(Cursor& cursor) {
    DiskBlobHash hash;
    hash.low  = cursor.read<std::uint64_t>();
    hash.high = cursor.read<std::uint64_t>();
    return hash;
}

} // namespace

std::size_t DiskRecordKeyHash::operator()(const DiskRecordKey& key) const noexcept {
    std::uint64_t hash = mix64(key.digest_low);
    hash ^= mix64(key.digest_high + 0x9e3779b97f4a7c15ULL);
    hash ^= mix64((static_cast<std::uint64_t>(key.frontier) << 32U) | key.identity_tag);
    return static_cast<std::size_t>(hash);
}

DiskBlobHash disk_blob_hash(std::span<const std::byte> payload) noexcept {
    DiskBlobHash hash;
    hash.low  = mix64(fnv1a64(payload) ^ payload.size());
    hash.high = mix64(fnv1a64(payload, 0x84222325cbf29ce4ULL) + 0x9e3779b97f4a7c15ULL +
                      (payload.size() << 1U));
    return hash;
}

ContextDiskStore::ContextDiskStore(ContextDiskStoreConfig config) : config_(std::move(config)) {
    if (config_.signature.empty() || config_.signature.size() > kMaxSignatureBytes) {
        throw std::invalid_argument("context disk store signature is empty or too long");
    }
    if (config_.eviction_low_watermark_percent == 0 ||
        config_.eviction_low_watermark_percent > 100) {
        throw std::invalid_argument("context disk store low watermark must be within 1..100");
    }
    index_path_ = config_.directory / "index.log";
    blobs_path_ = config_.directory / "blobs.dat";
    std::filesystem::create_directories(config_.directory);
    open_store();
}

ContextDiskStore::~ContextDiskStore() = default;

void ContextDiskStore::reset_store() {
    index_.close();
    blobs_.close();
    blob_index_.clear();
    records_.clear();
    live_bytes_ = 0;
    dead_bytes_ = 0;
    dead_blobs_ = 0;
    std::error_code code;
    std::filesystem::remove(index_path_, code);
    std::filesystem::remove(blobs_path_, code);

    index_ = core::PositionalFile(index_path_, core::FileMode::CreateOrOpen,
                                  core::FileAccess::ReadWrite);
    blobs_ = core::PositionalFile(blobs_path_, core::FileMode::CreateOrOpen,
                                  core::FileAccess::ReadWrite);

    core::AlignedBuffer header(static_cast<std::size_t>(kHeaderBytes));
    std::memset(header.data(), 0, header.size());
    std::vector<std::byte> fields;
    put_u64(fields, kStoreMagic);
    put_u32(fields, kStoreFormat);
    put_u32(fields, static_cast<std::uint32_t>(kHeaderBytes));
    put_u32(fields, static_cast<std::uint32_t>(config_.signature.size()));
    put_u32(fields, 0);
    std::memcpy(header.data(), fields.data(), fields.size());
    std::memcpy(header.data() + fields.size(), config_.signature.data(),
                config_.signature.size());
    index_.write_at(0, header.span());
    index_.truncate(kHeaderBytes);
    index_.flush();
    blobs_.truncate(0);
    blobs_.flush();
    core::sync_directory(config_.directory);
}

void ContextDiskStore::open_store() {
    const bool present =
        std::filesystem::exists(index_path_) && std::filesystem::exists(blobs_path_);
    if (!present) {
        reset_store();
        return;
    }
    index_ = core::PositionalFile(index_path_, core::FileMode::CreateOrOpen,
                                  core::FileAccess::ReadWrite);
    blobs_ = core::PositionalFile(blobs_path_, core::FileMode::CreateOrOpen,
                                  core::FileAccess::ReadWrite);
    if (index_.bytes() < kHeaderBytes) {
        reset_store();
        return;
    }
    core::AlignedBuffer header(static_cast<std::size_t>(kHeaderBytes));
    index_.read_exact(0, header.span());
    Cursor cursor(header.span());
    const auto magic      = cursor.read<std::uint64_t>();
    const auto format     = cursor.read<std::uint32_t>();
    const auto bytes      = cursor.read<std::uint32_t>();
    const auto signature  = cursor.read<std::uint32_t>();
    (void)cursor.read<std::uint32_t>();
    // A foreign or superseded store is discarded rather than partially trusted: matching a record
    // written by a different artifact or KV layout would restore semantically wrong state.
    if (cursor.truncated() || magic != kStoreMagic || format != kStoreFormat ||
        bytes != kHeaderBytes || signature != config_.signature.size() ||
        signature > kMaxSignatureBytes ||
        std::memcmp(header.data() + cursor.position(), config_.signature.data(), signature) != 0) {
        reset_store();
        return;
    }
    replay_index(kHeaderBytes);
}

void ContextDiskStore::replay_index(std::uint64_t payload_offset) {
    const std::uint64_t total = index_.bytes();
    std::vector<std::byte> log(static_cast<std::size_t>(total - payload_offset));
    if (!log.empty()) { index_.read_exact(payload_offset, log); }

    std::size_t position  = 0;
    std::size_t committed = 0;
    while (position + kEntryHeaderBytes <= log.size()) {
        Cursor header(std::span<const std::byte>(log).subspan(position, kEntryHeaderBytes));
        const auto payload_bytes = header.read<std::uint32_t>();
        const auto kind          = header.read<std::uint16_t>();
        (void)header.read<std::uint16_t>();
        const auto checksum = header.read<std::uint64_t>();
        const std::size_t padded =
            (static_cast<std::size_t>(payload_bytes) + 7U) & ~static_cast<std::size_t>(7U);
        if (position + kEntryHeaderBytes + padded > log.size()) { break; }
        const std::span<const std::byte> payload(log.data() + position + kEntryHeaderBytes,
                                                 payload_bytes);
        if (fnv1a64(payload) != checksum) { break; }

        Cursor cursor(payload);
        switch (kind) {
        case kEntryBlobCommit: {
            const DiskBlobHash hash   = take_hash(cursor);
            const std::uint64_t start = cursor.read<std::uint64_t>();
            const std::uint64_t size  = cursor.read<std::uint64_t>();
            const std::uint64_t aligned = core::align_up_direct(size);
            if (cursor.truncated() || size == 0 || start + aligned > blobs_.bytes()) { break; }
            BlobExtent& extent = blob_index_[hash];
            if (extent.bytes == 0) {
                extent = BlobExtent{.offset = start, .bytes = size, .aligned = aligned,
                                    .refcount = 0};
                dead_bytes_ += aligned;
                ++dead_blobs_;
            }
            break;
        }
        case kEntryRecordPublish: {
            const DiskRecordKey key     = take_key(cursor);
            const auto sequence         = cursor.read<std::uint64_t>();
            const auto page_bytes       = cursor.read<std::uint32_t>();
            const auto page_count       = cursor.read<std::uint32_t>();
            const auto state_bytes      = cursor.read<std::uint32_t>();
            (void)cursor.read<std::uint32_t>();
            RecordEntry entry;
            entry.descriptor = DiskRecordDescriptor{.key         = key,
                                                    .page_bytes  = page_bytes,
                                                    .page_count  = page_count,
                                                    .state_bytes = state_bytes,
                                                    .payload_bytes = 0,
                                                    .sequence      = sequence};
            entry.has_state  = state_bytes != 0;
            if (entry.has_state) { entry.state_hash = take_hash(cursor); }
            entry.page_hashes.reserve(page_count);
            for (std::uint32_t page = 0; page < page_count; ++page) {
                entry.page_hashes.push_back(take_hash(cursor));
            }
            if (cursor.truncated()) { break; }
            entry.access_sequence = sequence;
            publish_clock_        = std::max(publish_clock_, sequence);
            access_clock_         = std::max(access_clock_, sequence);
            const auto existing   = records_.find(key);
            if (existing != records_.end()) { erase_record_locked(key, false); }
            bool resolvable = true;
            if (entry.has_state && !blob_index_.contains(entry.state_hash)) { resolvable = false; }
            for (const DiskBlobHash& hash : entry.page_hashes) {
                if (!blob_index_.contains(hash)) { resolvable = false; }
            }
            if (!resolvable) { break; }
            if (entry.has_state) {
                retain_blob(entry.state_hash);
                entry.descriptor.payload_bytes += state_bytes;
            }
            for (const DiskBlobHash& hash : entry.page_hashes) {
                retain_blob(hash);
                entry.descriptor.payload_bytes += page_bytes;
            }
            records_.emplace(key, std::move(entry));
            break;
        }
        case kEntryRecordDrop: {
            const DiskRecordKey key = take_key(cursor);
            if (cursor.truncated()) { break; }
            erase_record_locked(key, false);
            break;
        }
        case kEntryRecordTouch: {
            const DiskRecordKey key    = take_key(cursor);
            const std::uint64_t access = cursor.read<std::uint64_t>();
            if (cursor.truncated()) { break; }
            const auto found           = records_.find(key);
            if (found != records_.end()) { found->second.access_sequence = access; }
            access_clock_ = std::max(access_clock_, access);
            break;
        }
        default:
            break;
        }
        position += kEntryHeaderBytes + padded;
        committed = position;
    }
    // A torn tail is discarded so the next append starts from a consistent journal.
    if (committed != log.size()) { index_.truncate(payload_offset + committed); }
}

void ContextDiskStore::append_entry(std::uint16_t kind, std::span<const std::byte> payload) {
    std::vector<std::byte> framed;
    framed.reserve(kEntryHeaderBytes + payload.size() + 8U);
    put_u32(framed, static_cast<std::uint32_t>(payload.size()));
    const auto offset = framed.size();
    framed.resize(offset + 4U);
    std::memcpy(framed.data() + offset, &kind, sizeof(kind));
    const std::uint16_t reserved = 0;
    std::memcpy(framed.data() + offset + 2U, &reserved, sizeof(reserved));
    put_u64(framed, fnv1a64(payload));
    framed.insert(framed.end(), payload.begin(), payload.end());
    framed.resize((framed.size() + 7U) & ~static_cast<std::size_t>(7U));
    index_.append(framed);
}

void ContextDiskStore::retain_blob(const DiskBlobHash& hash) {
    const auto found = blob_index_.find(hash);
    if (found == blob_index_.end()) { return; }
    if (found->second.refcount == 0) {
        live_bytes_ += found->second.aligned;
        dead_bytes_ -= std::min(dead_bytes_, found->second.aligned);
        if (dead_blobs_ != 0) { --dead_blobs_; }
    }
    ++found->second.refcount;
}

void ContextDiskStore::release_blob(const DiskBlobHash& hash) {
    const auto found = blob_index_.find(hash);
    if (found == blob_index_.end() || found->second.refcount == 0) { return; }
    if (--found->second.refcount == 0) {
        live_bytes_ -= std::min(live_bytes_, found->second.aligned);
        dead_bytes_ += found->second.aligned;
        ++dead_blobs_;
    }
}

void ContextDiskStore::erase_record_locked(const DiskRecordKey& key, bool journal) {
    const auto found = records_.find(key);
    if (found == records_.end()) { return; }
    if (found->second.has_state) { release_blob(found->second.state_hash); }
    for (const DiskBlobHash& hash : found->second.page_hashes) { release_blob(hash); }
    records_.erase(found);
    if (journal) {
        std::vector<std::byte> payload;
        put_key(payload, key);
        append_entry(kEntryRecordDrop, payload);
    }
}

DiskBlobHash ContextDiskStore::append_blob(std::span<const std::byte> payload,
                                           std::vector<DiskBlobHash>& appended) {
    const DiskBlobHash hash = disk_blob_hash(payload);
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto found = blob_index_.find(hash);
    if (found != blob_index_.end() && found->second.bytes == payload.size()) { return hash; }

    const std::uint64_t aligned = core::align_up_direct(payload.size());
    if (staging_.size() < aligned) { staging_.resize(static_cast<std::size_t>(aligned)); }
    std::memcpy(staging_.data(), payload.data(), payload.size());
    std::memset(staging_.data() + payload.size(),
                0, static_cast<std::size_t>(aligned) - payload.size());
    const std::uint64_t offset = core::align_up_direct(blobs_.bytes());
    blobs_.write_at(offset, staging_.span().first(static_cast<std::size_t>(aligned)));
    stats_.bytes_written += aligned;

    blob_index_[hash] =
        BlobExtent{.offset = offset, .bytes = payload.size(), .aligned = aligned, .refcount = 0};
    dead_bytes_ += aligned;
    ++dead_blobs_;
    appended.push_back(hash);

    std::vector<std::byte> entry;
    put_hash(entry, hash);
    put_u64(entry, offset);
    put_u64(entry, payload.size());
    append_entry(kEntryBlobCommit, entry);
    return hash;
}

ContextDiskStore::RecordWriter::RecordWriter(ContextDiskStore& store, DiskRecordKey key,
                                             std::uint32_t page_bytes)
    : store_(&store), key_(key), page_bytes_(page_bytes) {}

ContextDiskStore::RecordWriter::RecordWriter(RecordWriter&& other) noexcept
    : store_(std::exchange(other.store_, nullptr)), key_(other.key_),
      page_bytes_(other.page_bytes_), state_bytes_(other.state_bytes_),
      has_state_(other.has_state_), finished_(std::exchange(other.finished_, true)),
      page_hashes_(std::move(other.page_hashes_)), state_hash_(other.state_hash_),
      appended_(std::move(other.appended_)) {}

ContextDiskStore::RecordWriter::~RecordWriter() { abandon(); }

void ContextDiskStore::RecordWriter::abandon() noexcept {
    // Extents already appended stay in the pool unreferenced; the sweep reclaims them. Nothing
    // is deleted here, because the write cost is already spent either way.
    finished_ = true;
    store_    = nullptr;
}

void ContextDiskStore::RecordWriter::write_state_image(std::span<const std::byte> payload) {
    if (store_ == nullptr || finished_) { throw std::logic_error("disk record writer is closed"); }
    if (payload.empty()) { return; }
    state_hash_  = store_->append_blob(payload, appended_);
    state_bytes_ = static_cast<std::uint32_t>(payload.size());
    has_state_   = true;
}

void ContextDiskStore::RecordWriter::write_page(std::span<const std::byte> payload) {
    if (store_ == nullptr || finished_) { throw std::logic_error("disk record writer is closed"); }
    if (payload.size() != page_bytes_) {
        throw std::invalid_argument("disk record page size does not match the record geometry");
    }
    page_hashes_.push_back(store_->append_blob(payload, appended_));
}

bool ContextDiskStore::RecordWriter::publish() {
    if (store_ == nullptr || finished_) { return false; }
    ContextDiskStore* const store = store_;
    const bool published          = store->publish_record(*this);
    finished_                     = true;
    store_                        = nullptr;
    return published;
}

bool ContextDiskStore::publish_record(RecordWriter& writer) {
    // Publication order is what makes a torn write safe: the extents are durable first, and only
    // then does the journal gain the entry that names them.
    blobs_.flush();

    std::vector<std::byte> payload;
    const std::uint64_t sequence = ++publish_clock_;
    put_key(payload, writer.key_);
    put_u64(payload, sequence);
    put_u32(payload, writer.page_bytes_);
    put_u32(payload, static_cast<std::uint32_t>(writer.page_hashes_.size()));
    put_u32(payload, writer.state_bytes_);
    put_u32(payload, 0);
    if (writer.has_state_) { put_hash(payload, writer.state_hash_); }
    for (const DiskBlobHash& hash : writer.page_hashes_) { put_hash(payload, hash); }

    const std::lock_guard<std::mutex> guard(mutex_);
    append_entry(kEntryRecordPublish, payload);
    index_.flush();

    erase_record_locked(writer.key_, false);
    RecordEntry entry;
    entry.descriptor =
        DiskRecordDescriptor{.key           = writer.key_,
                             .page_bytes    = writer.page_bytes_,
                             .page_count    = static_cast<std::uint32_t>(writer.page_hashes_.size()),
                             .state_bytes   = writer.state_bytes_,
                             .payload_bytes = 0,
                             .sequence      = sequence};
    entry.has_state  = writer.has_state_;
    entry.state_hash = writer.state_hash_;
    entry.page_hashes = writer.page_hashes_;
    if (entry.has_state) {
        retain_blob(entry.state_hash);
        entry.descriptor.payload_bytes += writer.state_bytes_;
    }
    for (const DiskBlobHash& hash : entry.page_hashes) {
        retain_blob(hash);
        entry.descriptor.payload_bytes += writer.page_bytes_;
    }
    entry.access_sequence = ++access_clock_;
    records_.insert_or_assign(writer.key_, std::move(entry));
    ++stats_.records_published;
    return true;
}

ContextDiskStore::RecordWriter ContextDiskStore::begin_record(DiskRecordKey key,
                                                              std::uint32_t page_bytes) {
    return RecordWriter(*this, key, page_bytes);
}

std::optional<DiskRecordDescriptor> ContextDiskStore::lookup(const DiskRecordKey& key) {
    const std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.lookups;
    const auto found = records_.find(key);
    if (found == records_.end()) { return std::nullopt; }
    found->second.access_sequence = ++access_clock_;
    ++stats_.hits;
    return found->second.descriptor;
}

std::optional<DiskRecordDescriptor>
ContextDiskStore::lookup_longest(std::span<const DiskRecordKey> candidates) {
    const std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.lookups;
    std::optional<DiskRecordDescriptor> best;
    RecordEntry* chosen = nullptr;
    for (const DiskRecordKey& key : candidates) {
        const auto found = records_.find(key);
        if (found == records_.end()) { continue; }
        if (!best || found->second.descriptor.key.frontier > best->key.frontier) {
            best   = found->second.descriptor;
            chosen = &found->second;
        }
    }
    if (chosen != nullptr) {
        chosen->access_sequence = ++access_clock_;
        ++stats_.hits;
    }
    return best;
}

void ContextDiskStore::read_extent(const DiskBlobHash& hash, std::span<std::byte> destination) {
    BlobExtent extent;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        const auto found = blob_index_.find(hash);
        if (found == blob_index_.end()) {
            throw std::runtime_error("context disk store extent is missing");
        }
        extent = found->second;
    }
    if (destination.size() < extent.bytes) {
        throw std::invalid_argument("context disk store destination is too small");
    }
    if (core::PositionalFile::direct_eligible(extent.offset, destination.data(),
                                              static_cast<std::size_t>(extent.aligned)) &&
        destination.size() >= extent.aligned) {
        blobs_.read_exact(extent.offset,
                          destination.first(static_cast<std::size_t>(extent.aligned)));
    } else {
        blobs_.read_exact(extent.offset, destination.first(static_cast<std::size_t>(extent.bytes)));
    }
    const std::lock_guard<std::mutex> guard(mutex_);
    stats_.bytes_read += extent.bytes;
}

void ContextDiskStore::read_state_image(const DiskRecordDescriptor& record,
                                        std::span<std::byte> destination) {
    DiskBlobHash hash;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        const auto found = records_.find(record.key);
        if (found == records_.end() || !found->second.has_state) {
            throw std::runtime_error("context disk record has no state image");
        }
        hash = found->second.state_hash;
    }
    read_extent(hash, destination);
}

void ContextDiskStore::read_page(const DiskRecordDescriptor& record, std::uint32_t page_index,
                                 std::span<std::byte> destination) {
    DiskBlobHash hash;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        const auto found = records_.find(record.key);
        if (found == records_.end() || page_index >= found->second.page_hashes.size()) {
            throw std::runtime_error("context disk record page is out of range");
        }
        hash = found->second.page_hashes[page_index];
    }
    read_extent(hash, destination);
}

void ContextDiskStore::drop(const DiskRecordKey& key) {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (!records_.contains(key)) { return; }
    erase_record_locked(key, true);
    index_.flush();
    ++stats_.records_evicted;
}

void ContextDiskStore::enforce_capacity() {
    std::vector<std::pair<std::uint64_t, DiskRecordKey>> order;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (live_bytes_ <= config_.max_bytes) { return; }
        order.reserve(records_.size());
        for (const auto& [key, entry] : records_) {
            order.emplace_back(entry.access_sequence, key);
        }
    }
    std::sort(order.begin(), order.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });
    const std::uint64_t target =
        config_.max_bytes / 100ULL * config_.eviction_low_watermark_percent;
    const std::lock_guard<std::mutex> guard(mutex_);
    for (const auto& [sequence, key] : order) {
        if (live_bytes_ <= target) { break; }
        if (!records_.contains(key)) { continue; }
        erase_record_locked(key, true);
        ++stats_.records_evicted;
    }
    index_.flush();
}

bool ContextDiskStore::compaction_due() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    if (dead_blobs_ < config_.compaction_min_dead_blobs) { return false; }
    if (dead_bytes_ < config_.compaction_min_dead_bytes) { return false; }
    const std::uint64_t total = live_bytes_ + dead_bytes_;
    if (total == 0) { return false; }
    return dead_bytes_ * 100ULL >= total * config_.compaction_min_dead_percent;
}

void ContextDiskStore::rewrite_index(const std::filesystem::path& path,
                                     const std::vector<std::pair<DiskBlobHash, BlobExtent>>& blobs) {
    core::PositionalFile fresh(path, core::FileMode::CreateOrOpen, core::FileAccess::ReadWrite);
    fresh.truncate(0);
    core::AlignedBuffer header(static_cast<std::size_t>(kHeaderBytes));
    std::memset(header.data(), 0, header.size());
    std::vector<std::byte> fields;
    put_u64(fields, kStoreMagic);
    put_u32(fields, kStoreFormat);
    put_u32(fields, static_cast<std::uint32_t>(kHeaderBytes));
    put_u32(fields, static_cast<std::uint32_t>(config_.signature.size()));
    put_u32(fields, 0);
    std::memcpy(header.data(), fields.data(), fields.size());
    std::memcpy(header.data() + fields.size(), config_.signature.data(), config_.signature.size());
    fresh.write_at(0, header.span());

    std::swap(index_, fresh);
    for (const auto& [hash, extent] : blobs) {
        std::vector<std::byte> entry;
        put_hash(entry, hash);
        put_u64(entry, extent.offset);
        put_u64(entry, extent.bytes);
        append_entry(kEntryBlobCommit, entry);
    }
    for (const auto& [key, record] : records_) {
        std::vector<std::byte> payload;
        put_key(payload, key);
        put_u64(payload, record.descriptor.sequence);
        put_u32(payload, record.descriptor.page_bytes);
        put_u32(payload, record.descriptor.page_count);
        put_u32(payload, record.descriptor.state_bytes);
        put_u32(payload, 0);
        if (record.has_state) { put_hash(payload, record.state_hash); }
        for (const DiskBlobHash& hash : record.page_hashes) { put_hash(payload, hash); }
        append_entry(kEntryRecordPublish, payload);

        std::vector<std::byte> touch;
        put_key(touch, key);
        put_u64(touch, record.access_sequence);
        append_entry(kEntryRecordTouch, touch);
    }
    index_.flush();
    std::swap(index_, fresh);
}

bool ContextDiskStore::compact(const DiskCancellationToken& cancel) {
    std::vector<std::pair<DiskBlobHash, BlobExtent>> live;
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        live.reserve(blob_index_.size());
        for (const auto& [hash, extent] : blob_index_) {
            if (extent.refcount != 0) { live.emplace_back(hash, extent); }
        }
    }
    std::sort(live.begin(), live.end(), [](const auto& left, const auto& right) {
        return left.second.offset < right.second.offset;
    });

    const std::filesystem::path blobs_tmp = blobs_path_.string() + ".tmp";
    const std::filesystem::path index_tmp = index_path_.string() + ".tmp";
    std::error_code code;
    std::filesystem::remove(blobs_tmp, code);
    std::filesystem::remove(index_tmp, code);

    {
        core::PositionalFile fresh(blobs_tmp, core::FileMode::CreateOrOpen,
                                   core::FileAccess::ReadWrite);
        core::AlignedBuffer batch;
        std::uint64_t cursor = 0;
        for (auto& [hash, extent] : live) {
            // The copy loop is where a compaction can be abandoned for free: no journal entry has
            // been written and the live store is untouched.
            if (cancel.requested()) {
                std::filesystem::remove(blobs_tmp, code);
                return false;
            }
            if (batch.size() < extent.aligned) {
                batch.resize(static_cast<std::size_t>(extent.aligned));
            }
            const auto span = batch.span().first(static_cast<std::size_t>(extent.aligned));
            blobs_.read_exact(extent.offset, span);
            fresh.write_at(cursor, span);
            extent.offset = cursor;
            cursor += extent.aligned;
        }
        fresh.truncate(cursor);
        fresh.flush();
    }

    std::unordered_map<DiskBlobHash, BlobExtent, DiskBlobHashHasher> rebuilt;
    rebuilt.reserve(live.size());
    for (const auto& [hash, extent] : live) { rebuilt.emplace(hash, extent); }

    const std::lock_guard<std::mutex> guard(mutex_);
    rewrite_index(index_tmp, live);
    index_.close();
    blobs_.close();
    core::atomic_replace_file(blobs_tmp, blobs_path_);
    core::atomic_replace_file(index_tmp, index_path_);
    index_ = core::PositionalFile(index_path_, core::FileMode::OpenExisting,
                                  core::FileAccess::ReadWrite);
    blobs_ = core::PositionalFile(blobs_path_, core::FileMode::OpenExisting,
                                  core::FileAccess::ReadWrite);
    blob_index_ = std::move(rebuilt);
    dead_bytes_ = 0;
    dead_blobs_ = 0;
    ++stats_.compactions;
    return true;
}

ContextDiskStoreStats ContextDiskStore::stats() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    ContextDiskStoreStats snapshot = stats_;
    snapshot.records               = records_.size();
    snapshot.blobs                 = blob_index_.size();
    snapshot.live_bytes            = live_bytes_;
    snapshot.dead_bytes            = dead_bytes_;
    snapshot.dead_blobs            = dead_blobs_;
    snapshot.file_bytes            = blobs_.bytes() + index_.bytes();
    return snapshot;
}

} // namespace ninfer::runtime
