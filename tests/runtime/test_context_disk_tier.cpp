// Host-only coverage for the spill/restore state machine of the third context tier.
//
// The device half is a fake `ContextDiskTransferPort`: it holds page images in ordinary host
// memory and records every call, so the batching, the ordering of capture and publication, and
// the cancellation rule are all observable without a GPU.

#include "runtime/engine/context_cache/context_disk_tier.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
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
                ("ninfer_disk_tier_" + name + "_" + std::to_string(unique));
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

constexpr std::uint32_t kPageBytes  = 2048;
constexpr std::uint32_t kStateBytes = 512;

// Stand-in for a Program checkpoint: the pages and state image of one host state slot.
struct FakeCheckpoint {
    std::vector<std::byte> state;
    std::vector<std::vector<std::byte>> pages;
};

FakeCheckpoint make_checkpoint(std::uint64_t seed, std::uint32_t pages) {
    FakeCheckpoint checkpoint;
    std::mt19937_64 source(seed);
    checkpoint.state.resize(kStateBytes);
    for (std::size_t index = 0; index + 8 <= kStateBytes; index += 8) {
        const std::uint64_t value = source();
        std::memcpy(checkpoint.state.data() + index, &value, sizeof(value));
    }
    for (std::uint32_t page = 0; page < pages; ++page) {
        std::vector<std::byte> payload(kPageBytes);
        for (std::size_t index = 0; index + 8 <= kPageBytes; index += 8) {
            const std::uint64_t value = source();
            std::memcpy(payload.data() + index, &value, sizeof(value));
        }
        checkpoint.pages.push_back(std::move(payload));
    }
    return checkpoint;
}

class FakePort final : public ContextDiskTransferPort {
public:
    explicit FakePort(std::uint32_t batch) : batch_(batch) {}

    // --- test-facing state -------------------------------------------------------------------
    std::map<std::uint64_t, FakeCheckpoint> slots;      // host state slots available to capture
    std::map<std::uint64_t, FakeCheckpoint> restored;   // destinations filled by a restore
    std::uint64_t restore_destination = 0;
    std::uint32_t largest_capture_batch = 0;
    std::uint32_t largest_restore_batch = 0;
    std::uint64_t captures_ended_published = 0;
    std::uint64_t captures_ended_unpublished = 0;
    bool fail_next_restore_commit = false;
    bool refuse_next_restore      = false;

    [[nodiscard]] std::uint32_t batch_pages() const noexcept override { return batch_; }

    [[nodiscard]] std::optional<DiskCheckpointGeometry>
    begin_capture(std::uint64_t owner_token) override {
        const auto found = slots.find(owner_token);
        if (found == slots.end()) { return std::nullopt; }
        capturing_ = &found->second;
        return DiskCheckpointGeometry{
            .page_bytes  = kPageBytes,
            .page_count  = static_cast<std::uint32_t>(found->second.pages.size()),
            .state_bytes = static_cast<std::uint32_t>(found->second.state.size())};
    }

    [[nodiscard]] std::span<const std::byte> capture_state_image() override {
        if (capturing_ == nullptr) { return {}; }
        return capturing_->state;
    }

    [[nodiscard]] std::span<const std::byte> capture_pages(std::uint32_t first,
                                                           std::uint32_t count) override {
        if (capturing_ == nullptr || first + count > capturing_->pages.size()) { return {}; }
        require(count <= batch_, "the tier staged more pages than the port allows");
        largest_capture_batch = std::max(largest_capture_batch, count);
        staging_.assign(static_cast<std::size_t>(count) * kPageBytes, std::byte{});
        for (std::uint32_t page = 0; page < count; ++page) {
            std::memcpy(staging_.data() + static_cast<std::size_t>(page) * kPageBytes,
                        capturing_->pages[first + page].data(), kPageBytes);
        }
        return staging_;
    }

    void end_capture(bool published) noexcept override {
        capturing_ = nullptr;
        if (published) {
            ++captures_ended_published;
        } else {
            ++captures_ended_unpublished;
        }
    }

    [[nodiscard]] std::optional<DiskCheckpointGeometry>
    begin_restore(const DiskRecordDescriptor& record) override {
        if (refuse_next_restore) {
            refuse_next_restore = false;
            return std::nullopt;
        }
        target_ = &restored[restore_destination];
        target_->state.clear();
        target_->pages.assign(record.page_count, {});
        return DiskCheckpointGeometry{.page_bytes  = record.page_bytes,
                                      .page_count  = record.page_count,
                                      .state_bytes = record.state_bytes};
    }

    [[nodiscard]] std::span<std::byte> restore_staging(std::uint32_t count) override {
        require(count <= batch_, "the tier staged more pages than the port allows");
        largest_restore_batch = std::max(largest_restore_batch, count);
        staging_.assign(static_cast<std::size_t>(count) * kPageBytes, std::byte{});
        return staging_;
    }

    [[nodiscard]] bool commit_pages(std::uint32_t first, std::uint32_t count) override {
        if (fail_next_restore_commit) {
            fail_next_restore_commit = false;
            return false;
        }
        if (target_ == nullptr) { return false; }
        for (std::uint32_t page = 0; page < count; ++page) {
            target_->pages[first + page].assign(
                staging_.begin() + static_cast<std::ptrdiff_t>(page) * kPageBytes,
                staging_.begin() + static_cast<std::ptrdiff_t>(page + 1) * kPageBytes);
        }
        return true;
    }

    [[nodiscard]] bool commit_state_image(std::span<const std::byte> payload) override {
        if (target_ == nullptr) { return false; }
        target_->state.assign(payload.begin(), payload.end());
        return true;
    }

    void end_restore(bool) noexcept override { target_ = nullptr; }

private:
    std::uint32_t batch_ = 1;
    FakeCheckpoint* capturing_ = nullptr;
    FakeCheckpoint* target_    = nullptr;
    std::vector<std::byte> staging_;
};

ContextDiskStoreConfig config_for(const std::filesystem::path& directory) {
    ContextDiskStoreConfig config;
    config.directory = directory;
    config.signature = "artifact:tier-test/kv:bf16";
    config.max_bytes = 1ULL << 30;
    return config;
}

DiskRecordKey key_for(std::uint32_t frontier) {
    return DiskRecordKey{.digest_low   = 0xabcd'0000ULL + frontier,
                         .digest_high  = 0x1234'0000ULL + frontier * 3U,
                         .frontier     = frontier,
                         .identity_tag = 1};
}

void expect_equal(const FakeCheckpoint& left, const FakeCheckpoint& right, const char* message) {
    require(left.state == right.state, message);
    require(left.pages.size() == right.pages.size(), message);
    for (std::size_t page = 0; page < left.pages.size(); ++page) {
        require(left.pages[page] == right.pages[page], message);
    }
}

// The whole hook sequence: evicting a host slot spills it asynchronously, and a later prefix
// lookup that misses Device and Host finds the record and restores it.
void test_spill_then_restore() {
    TempDirectory directory("spill_restore");
    FakePort port(8);
    port.slots[1] = make_checkpoint(7, 37);

    ContextDiskTier tier(config_for(directory.path()), port);
    tier.request_spill(DiskSpillRequest{.key = key_for(2368), .owner_token = 1});
    tier.drain();

    const auto after_spill = tier.stats();
    require(after_spill.spills_published == 1, "the evicted slot was not spilled");
    require(port.captures_ended_published == 1, "the port was not told the capture published");
    require(after_spill.spilled_bytes ==
                static_cast<std::uint64_t>(37) * kPageBytes + kStateBytes,
            "the spilled byte count disagrees with the checkpoint size");
    require(port.largest_capture_batch == 8, "the capture was not batched at the port's bound");

    const std::vector<DiskRecordKey> candidates{key_for(1184), key_for(2368)};
    const auto found = tier.lookup(candidates);
    require(found.has_value(), "the spilled record was not found");
    require(found->key.frontier == 2368, "the wrong frontier was selected");

    port.restore_destination = 99;
    require(tier.restore(*found), "the restore failed");
    expect_equal(port.restored[99], port.slots[1], "the restored checkpoint differs");
    require(port.largest_restore_batch == 8, "the restore was not batched at the port's bound");

    const auto after_restore = tier.stats();
    require(after_restore.restores == 1 && after_restore.hits == 1,
            "the restore counters did not advance");
    require(after_restore.restored_bytes == after_spill.spilled_bytes,
            "restored bytes disagree with spilled bytes");
}

// A cold process finds records written by an earlier one, which is the whole point of the tier.
void test_restore_after_restart() {
    TempDirectory directory("restart");
    const FakeCheckpoint original = make_checkpoint(19, 21);
    {
        FakePort port(4);
        port.slots[5] = original;
        ContextDiskTier tier(config_for(directory.path()), port);
        tier.request_spill(DiskSpillRequest{.key = key_for(1344), .owner_token = 5});
        tier.drain();
        require(tier.stats().spills_published == 1, "the first process did not publish");
    }
    FakePort port(4);
    ContextDiskTier tier(config_for(directory.path()), port);
    const std::vector<DiskRecordKey> candidates{key_for(1344)};
    const auto found = tier.lookup(candidates);
    require(found.has_value(), "a restarted process did not see the published record");
    port.restore_destination = 1;
    require(tier.restore(*found), "the cold restore failed");
    expect_equal(port.restored[1], original, "the cold-restored checkpoint differs");
}

void test_missing_slot_and_refused_restore() {
    TempDirectory directory("refused");
    FakePort port(4);
    port.slots[1] = make_checkpoint(3, 5);
    ContextDiskTier tier(config_for(directory.path()), port);

    // The slot disappeared between the eviction decision and the spill.
    tier.request_spill(DiskSpillRequest{.key = key_for(320), .owner_token = 404});
    tier.drain();
    require(tier.stats().spills_published == 0 && tier.stats().spills_dropped == 1,
            "a spill of a vanished slot was published");
    require(!tier.lookup(std::vector<DiskRecordKey>{key_for(320)}).has_value(),
            "a dropped spill left a record behind");

    tier.request_spill(DiskSpillRequest{.key = key_for(320), .owner_token = 1});
    tier.drain();
    const auto found = tier.lookup(std::vector<DiskRecordKey>{key_for(320)});
    require(found.has_value(), "the second spill did not publish");

    // No destination could be reserved: admission must fall back to prefill.
    port.refuse_next_restore = true;
    require(!tier.restore(*found), "a refused restore reported success");
    require(tier.stats().restore_failures == 1, "a refused restore was not counted");

    // A failing device commit leaves the record intact for a later attempt.
    port.fail_next_restore_commit = true;
    port.restore_destination      = 2;
    require(!tier.restore(*found), "a failed page commit reported success");
    require(tier.lookup(std::vector<DiskRecordKey>{key_for(320)}).has_value(),
            "a failed restore retired the record");
    port.restore_destination = 3;
    require(tier.restore(*found), "the retry after a failed commit did not succeed");
    expect_equal(port.restored[3], port.slots[1], "the retried restore differs");
}

// Cancellation is a latency-priority signal, not an invalidation: it drops a spill that has not
// started and never discards one whose bytes are already written.
void test_request_priority_drops_only_unstarted_spills() {
    TempDirectory directory("priority");
    FakePort port(4);
    port.slots[1] = make_checkpoint(23, 9);
    port.slots[2] = make_checkpoint(29, 9);
    ContextDiskTier tier(config_for(directory.path()), port);

    tier.note_request_arrival();
    tier.request_spill(DiskSpillRequest{.key = key_for(576), .owner_token = 1});
    tier.drain();
    require(tier.stats().spills_dropped == 1 && tier.stats().spills_published == 0,
            "a spill queued under request priority still ran");
    require(port.captures_ended_published == 0 && port.captures_ended_unpublished == 0,
            "a dropped spill still captured from the port");

    tier.clear_request_priority();
    tier.request_spill(DiskSpillRequest{.key = key_for(576), .owner_token = 2});
    tier.drain();
    require(tier.stats().spills_published == 1, "the spill did not run after priority cleared");
}

// The queue is bounded, and a repeated request for the same prefix replaces rather than stacks.
void test_queue_bounds() {
    TempDirectory directory("queue");
    FakePort port(4);
    for (std::uint64_t slot = 1; slot <= 6; ++slot) {
        port.slots[slot] = make_checkpoint(40 + slot, 4);
    }
    ContextDiskTier tier(config_for(directory.path()), port, 2);
    tier.note_request_arrival(); // hold the worker off so the queue actually fills
    for (std::uint64_t slot = 1; slot <= 6; ++slot) {
        tier.request_spill(
            DiskSpillRequest{.key = key_for(static_cast<std::uint32_t>(64 * slot)),
                             .owner_token = slot});
    }
    tier.drain();
    const auto stats = tier.stats();
    require(stats.spills_requested == 6, "requests were not all counted");
    require(stats.spills_published == 0, "request priority did not hold the queue off");
    require(stats.spills_dropped >= 4, "the queue grew past its bound");
}

} // namespace

int main() {
    try {
        test_spill_then_restore();
        test_restore_after_restart();
        test_missing_slot_and_refused_restore();
        test_request_priority_drops_only_unstarted_spills();
        test_queue_bounds();
        std::cout << "ok\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
