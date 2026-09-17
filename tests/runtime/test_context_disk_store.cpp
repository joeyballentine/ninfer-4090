// Host-only coverage for the persistent context tier's on-disk format and capacity policy.
// Nothing here needs a device: a record is opaque bytes plus a prefix identity.

#include "runtime/engine/context_cache/context_disk_store.h"

#include <cstring>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::runtime;

namespace {

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}

class TempDirectory {
public:
    explicit TempDirectory(const std::string& name) {
        const auto unique = static_cast<unsigned long long>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        path_ = std::filesystem::temp_directory_path() /
                ("ninfer_disk_store_" + name + "_" + std::to_string(unique));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDirectory() {
        std::error_code code;
        std::filesystem::remove_all(path_, code);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

constexpr std::uint32_t kPageBytes = 4096;

std::vector<std::byte> pattern(std::uint64_t seed, std::size_t bytes) {
    std::vector<std::byte> payload(bytes);
    std::mt19937_64 source(seed);
    for (std::size_t index = 0; index + 8 <= bytes; index += 8) {
        const std::uint64_t value = source();
        std::memcpy(payload.data() + index, &value, sizeof(value));
    }
    return payload;
}

DiskRecordKey key_for(std::uint32_t frontier, std::uint64_t salt = 1) {
    return DiskRecordKey{.digest_low   = 0x1111'0000'0000'0000ULL + frontier * 7U + salt,
                         .digest_high  = 0x2222'0000'0000'0000ULL + frontier * 13U + salt,
                         .frontier     = frontier,
                         .identity_tag = 3};
}

ContextDiskStoreConfig config_for(const std::filesystem::path& directory, std::uint64_t max_bytes) {
    ContextDiskStoreConfig config;
    config.directory                 = directory;
    config.signature                 = "artifact:test/kv:bf16";
    config.max_bytes                 = max_bytes;
    config.compaction_min_dead_blobs = 4;
    config.compaction_min_dead_bytes = 16 * kPageBytes;
    config.compaction_min_dead_percent = 25;
    return config;
}

// Writes one record with `pages` distinct page payloads plus a state image, and publishes it.
void publish(ContextDiskStore& store, const DiskRecordKey& key, std::uint32_t pages,
             std::uint64_t seed) {
    auto writer = store.begin_record(key, kPageBytes);
    writer.write_state_image(pattern(seed, 777));
    for (std::uint32_t page = 0; page < pages; ++page) {
        writer.write_page(pattern(seed * 1000U + page, kPageBytes));
    }
    require(writer.publish(), "record publication failed");
}

void check_record(ContextDiskStore& store, const DiskRecordKey& key, std::uint32_t pages,
                  std::uint64_t seed) {
    const auto found = store.lookup(key);
    require(found.has_value(), "published record was not found");
    require(found->page_count == pages, "restored page count disagrees");
    require(found->state_bytes == 777, "restored state image size disagrees");
    std::vector<std::byte> buffer(kPageBytes);
    store.read_state_image(*found, buffer);
    const auto state = pattern(seed, 777);
    require(std::memcmp(buffer.data(), state.data(), state.size()) == 0,
            "restored state image bytes differ");
    for (std::uint32_t page = 0; page < pages; ++page) {
        store.read_page(*found, page, buffer);
        const auto expected = pattern(seed * 1000U + page, kPageBytes);
        require(std::memcmp(buffer.data(), expected.data(), kPageBytes) == 0,
                "restored page bytes differ");
    }
}

void test_round_trip_across_reopen() {
    TempDirectory directory("roundtrip");
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        publish(store, key_for(64), 3, 11);
        publish(store, key_for(128), 5, 12);
        check_record(store, key_for(64), 3, 11);
    }
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        require(store.stats().records == 2, "reopened store lost published records");
        check_record(store, key_for(64), 3, 11);
        check_record(store, key_for(128), 5, 12);
        require(!store.lookup(key_for(4096)).has_value(), "absent key reported a hit");
    }
}

// An extent stream that was written but whose publication entry never landed must not become a
// record: that is the donor's 8488278c failure mode inverted - abandoning after the flush loses
// the record, so the only safe rule is that the journal entry defines existence.
void test_unpublished_record_is_ignored() {
    TempDirectory directory("unpublished");
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        publish(store, key_for(64), 2, 21);
        auto writer = store.begin_record(key_for(256), kPageBytes);
        writer.write_state_image(pattern(99, 777));
        for (std::uint32_t page = 0; page < 4; ++page) {
            writer.write_page(pattern(900 + page, kPageBytes));
        }
        writer.abandon();
        require(!store.lookup(key_for(256)).has_value(),
                "an abandoned writer published a record");
    }
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        require(!store.lookup(key_for(256)).has_value(),
                "an abandoned record survived a reopen");
        check_record(store, key_for(64), 2, 21);
        require(store.stats().dead_blobs >= 4,
                "the abandoned extents were not accounted as reclaimable");
    }
}

// A torn journal tail - the crash window between appending an entry and flushing it - is
// discarded, and the records before it stay usable.
void test_torn_journal_tail_is_discarded() {
    TempDirectory directory("torn");
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        publish(store, key_for(64), 2, 31);
        publish(store, key_for(128), 2, 32);
    }
    {
        const std::filesystem::path index = directory.path() / "index.log";
        const auto bytes                  = std::filesystem::file_size(index);
        std::filesystem::resize_file(index, bytes - 24);
    }
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        check_record(store, key_for(64), 2, 31);
        require(!store.lookup(key_for(128)).has_value(),
                "a record whose journal entry was torn was still published");
    }
}

void test_signature_mismatch_is_ignored() {
    TempDirectory directory("signature");
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        publish(store, key_for(64), 3, 41);
    }
    {
        auto config      = config_for(directory.path(), 1ULL << 30);
        config.signature = "artifact:other/kv:int8";
        ContextDiskStore store(config);
        require(store.stats().records == 0, "a foreign signature matched a stored record");
        require(!store.lookup(key_for(64)).has_value(),
                "a record written under a different config signature was restorable");
        publish(store, key_for(64), 1, 42);
        check_record(store, key_for(64), 1, 42);
    }
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        require(store.stats().records == 0,
                "the original signature matched a store rewritten by another configuration");
    }
}

// Identical page payloads across turns share one extent, which is what keeps a per-turn spill a
// small delta write instead of a full rewrite.
void test_shared_extents_across_turns() {
    TempDirectory directory("cow");
    ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
    auto first = store.begin_record(key_for(64), kPageBytes);
    for (std::uint32_t page = 0; page < 8; ++page) {
        first.write_page(pattern(500 + page, kPageBytes));
    }
    require(first.publish(), "first turn did not publish");
    const auto after_first = store.stats();

    auto second = store.begin_record(key_for(128), kPageBytes);
    for (std::uint32_t page = 0; page < 10; ++page) {
        second.write_page(pattern(500 + page, kPageBytes));
    }
    require(second.publish(), "second turn did not publish");
    const auto after_second = store.stats();

    require(after_second.blobs == after_first.blobs + 2,
            "the shared prefix pages were written a second time");
    require(after_second.bytes_written - after_first.bytes_written == 2ULL * kPageBytes,
            "the delta write was larger than the two new pages");
}

void test_capacity_eviction_is_least_recently_used() {
    TempDirectory directory("lru");
    // Ten pages per record; cap at roughly four records so eviction has to choose.
    auto config                           = config_for(directory.path(), 40ULL * kPageBytes);
    config.eviction_low_watermark_percent = 60;
    ContextDiskStore store(config);
    for (std::uint32_t record = 0; record < 4; ++record) {
        publish(store, key_for(64 * (record + 1), record), 10, 100 + record);
    }
    require(store.stats().records == 4, "setup did not publish four records");
    // Touch the oldest record so recency, not publication order, decides the victim.
    require(store.lookup(key_for(64, 0)).has_value(), "the oldest record was already gone");

    publish(store, key_for(512, 9), 10, 200);
    store.enforce_capacity();

    const auto after = store.stats();
    require(after.live_bytes <= config.max_bytes / 100U * 60U,
            "eviction did not reach the low watermark");
    require(after.records_evicted == 3,
            "eviction did not retire the whole excess in one pass");
    require(store.lookup(key_for(64, 0)).has_value(), "the touched record was evicted first");
    require(!store.lookup(key_for(128, 1)).has_value(),
            "the least recently used record survived eviction");
    require(store.lookup(key_for(512, 9)).has_value(), "the newest record was evicted");
    require(after.records_evicted >= 1, "eviction was not counted");
}

void test_mark_and_sweep_compaction() {
    TempDirectory directory("compact");
    auto config = config_for(directory.path(), 1ULL << 30);
    ContextDiskStore store(config);
    for (std::uint32_t record = 0; record < 6; ++record) {
        publish(store, key_for(64 * (record + 1), record), 8, 300 + record);
    }
    // Retire half the records so their extents become reclaimable.
    for (std::uint32_t record = 0; record < 6; record += 2) {
        store.drop(key_for(64 * (record + 1), record));
    }
    const auto before = store.stats();
    require(before.dead_blobs >= 24, "dropping records did not produce reclaimable extents");
    require(store.compaction_due(), "the sweep guards rejected a clearly fragmented pool");

    DiskCancellationToken cancel;
    cancel.request();
    require(!store.compact(cancel), "a cancelled sweep reported success");
    require(store.stats().dead_blobs == before.dead_blobs,
            "a cancelled sweep mutated the live store");
    for (std::uint32_t record = 1; record < 6; record += 2) {
        check_record(store, key_for(64 * (record + 1), record), 8, 300 + record);
    }

    cancel.clear();
    require(store.compact(cancel), "the sweep failed");
    const auto after = store.stats();
    require(after.dead_blobs == 0 && after.dead_bytes == 0,
            "the sweep left reclaimable extents behind");
    require(after.file_bytes < before.file_bytes, "the sweep did not shrink the pool");
    require(after.records == 3, "the sweep changed the surviving record set");
    for (std::uint32_t record = 1; record < 6; record += 2) {
        check_record(store, key_for(64 * (record + 1), record), 8, 300 + record);
    }
    for (std::uint32_t record = 0; record < 6; record += 2) {
        require(!store.lookup(key_for(64 * (record + 1), record)).has_value(),
                "a dropped record reappeared after the sweep");
    }
    // The compacted files must reload with the rewritten offsets.
    publish(store, key_for(1024, 77), 4, 400);
}

void test_compacted_store_reloads() {
    TempDirectory directory("compact_reload");
    {
        ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
        for (std::uint32_t record = 0; record < 6; ++record) {
            publish(store, key_for(64 * (record + 1), record), 8, 500 + record);
        }
        for (std::uint32_t record = 0; record < 6; record += 2) {
            store.drop(key_for(64 * (record + 1), record));
        }
        DiskCancellationToken cancel;
        require(store.compact(cancel), "the sweep failed");
    }
    ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
    require(store.stats().records == 3, "the compacted journal lost records on reload");
    require(store.stats().dead_blobs == 0, "the compacted journal reloaded reclaimable extents");
    for (std::uint32_t record = 1; record < 6; record += 2) {
        check_record(store, key_for(64 * (record + 1), record), 8, 500 + record);
    }
}

// The extent locations a kernel-bypass DMA engine would be handed must describe the same bytes
// the portable read path returns.
void test_extent_locations_match_the_payload() {
    TempDirectory directory("extents");
    ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
    publish(store, key_for(192), 3, 71);
    const auto found = store.lookup(key_for(192));
    require(found.has_value(), "the record was not published");
    const auto extents = store.extent_locations(*found);
    require(extents.size() == 4, "the extent list does not cover the state image and every page");
    require(extents.front().bytes == 777, "the first extent is not the state image");

    ninfer::core::PositionalFile file(store.extent_file(), ninfer::core::FileMode::OpenExisting,
                                      ninfer::core::FileAccess::Read);
    std::vector<std::byte> direct(kPageBytes);
    std::vector<std::byte> through(kPageBytes);
    for (std::uint32_t page = 0; page < 3; ++page) {
        const auto& extent = extents[page + 1U];
        require(extent.bytes == kPageBytes, "a page extent has the wrong length");
        file.read_exact(extent.offset, std::span<std::byte>(direct).first(kPageBytes));
        store.read_page(*found, page, through);
        require(direct == through, "the extent location does not name the page's bytes");
    }
}

void test_longest_prefix_selection() {
    TempDirectory directory("longest");
    ContextDiskStore store(config_for(directory.path(), 1ULL << 30));
    publish(store, key_for(64), 1, 61);
    publish(store, key_for(256), 4, 62);
    publish(store, key_for(128), 2, 63);
    const std::vector<DiskRecordKey> candidates{key_for(64), key_for(128), key_for(256),
                                                key_for(512)};
    const auto best = store.lookup_longest(candidates);
    require(best.has_value(), "no candidate matched");
    require(best->key.frontier == 256, "the longest published prefix was not selected");
    const std::vector<DiskRecordKey> absent{key_for(512), key_for(1024)};
    require(!store.lookup_longest(absent).has_value(), "absent candidates reported a hit");
}

} // namespace

int main() {
    try {
        test_round_trip_across_reopen();
        test_unpublished_record_is_ignored();
        test_torn_journal_tail_is_discarded();
        test_signature_mismatch_is_ignored();
        test_shared_extents_across_turns();
        test_capacity_eviction_is_least_recently_used();
        test_mark_and_sweep_compaction();
        test_compacted_store_reloads();
        test_extent_locations_match_the_payload();
        test_longest_prefix_selection();
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
