#include "runtime/engine/context_cache/context_disk_tier.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime {

ContextDiskTier::ContextDiskTier(ContextDiskStoreConfig config, ContextDiskTransferPort& port,
                                 std::uint32_t queue_depth)
    : store_(std::move(config)), port_(port), queue_depth_(std::max(1U, queue_depth)) {
    worker_ = std::thread([this] { worker(); });
}

ContextDiskTier::~ContextDiskTier() {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        running_ = false;
        queue_.clear();
    }
    cancel_.request();
    wake_.notify_all();
    if (worker_.joinable()) { worker_.join(); }
}

void ContextDiskTier::request_spill(const DiskSpillRequest& request) {
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (!running_) { return; }
        ++stats_.spills_requested;
        // Replacing an existing request for the same prefix keeps the queue from holding two
        // captures of the same frontier.
        const auto duplicate = std::find_if(queue_.begin(), queue_.end(),
                                            [&](const DiskSpillRequest& pending) {
                                                return pending.key == request.key;
                                            });
        if (duplicate != queue_.end()) {
            *duplicate = request;
        } else {
            if (queue_.size() >= queue_depth_) {
                queue_.pop_front();
                ++stats_.spills_dropped;
            }
            queue_.push_back(request);
        }
    }
    wake_.notify_one();
}

void ContextDiskTier::note_request_arrival() noexcept { cancel_.request(); }

void ContextDiskTier::worker() {
    for (;;) {
        DiskSpillRequest request;
        {
            std::unique_lock<std::mutex> guard(mutex_);
            idle_.notify_all();
            wake_.wait(guard, [this] { return !running_ || !queue_.empty(); });
            if (!running_) { return; }
            request = queue_.front();
            queue_.pop_front();
            busy_ = true;
        }
        run_spill(request);
        {
            const std::lock_guard<std::mutex> guard(mutex_);
            busy_ = false;
        }
        idle_.notify_all();
    }
}

void ContextDiskTier::run_spill(const DiskSpillRequest& request) {
    // The only cancellation point that saves real work: nothing has been captured or written.
    // Once the extents are on disk the abort would perform the same I/O and then discard the
    // result, which is the failure the donor fixed in 8488278c.
    if (cancel_.requested()) {
        const std::lock_guard<std::mutex> guard(mutex_);
        ++stats_.spills_dropped;
        return;
    }

    const std::optional<DiskCheckpointGeometry> geometry = port_.begin_capture(request.owner_token);
    if (!geometry || geometry->page_bytes == 0) {
        const std::lock_guard<std::mutex> guard(mutex_);
        ++stats_.spills_dropped;
        return;
    }

    bool published    = false;
    std::uint64_t bytes = 0;
    try {
        ContextDiskStore::RecordWriter writer =
            store_.begin_record(request.key, geometry->page_bytes);
        {
            const std::lock_guard<std::mutex> io(io_mutex_);
            if (geometry->state_bytes != 0) {
                const std::span<const std::byte> image = port_.capture_state_image();
                if (image.size() != geometry->state_bytes) {
                    port_.end_capture(false);
                    const std::lock_guard<std::mutex> guard(mutex_);
                    ++stats_.spills_dropped;
                    return;
                }
                writer.write_state_image(image);
                bytes += image.size();
            }
        }
        const std::uint32_t batch = std::max(1U, port_.batch_pages());
        for (std::uint32_t first = 0; first < geometry->page_count; first += batch) {
            const std::uint32_t count = std::min(batch, geometry->page_count - first);
            // The I/O lock is released between batches so a restore on the admission path waits
            // at most one batch behind a spill.
            const std::lock_guard<std::mutex> io(io_mutex_);
            const std::span<const std::byte> staged = port_.capture_pages(first, count);
            if (staged.size() != static_cast<std::size_t>(count) * geometry->page_bytes) {
                port_.end_capture(false);
                const std::lock_guard<std::mutex> guard(mutex_);
                ++stats_.spills_dropped;
                return;
            }
            for (std::uint32_t page = 0; page < count; ++page) {
                writer.write_page(
                    staged.subspan(static_cast<std::size_t>(page) * geometry->page_bytes,
                                   geometry->page_bytes));
            }
            bytes += staged.size();
        }
        {
            const std::lock_guard<std::mutex> io(io_mutex_);
            published = writer.publish();
        }
    } catch (const std::exception&) {
        // A store-level I/O failure retires this spill; the tier stays usable for the next one.
        port_.end_capture(false);
        const std::lock_guard<std::mutex> guard(mutex_);
        ++stats_.spills_dropped;
        return;
    }
    port_.end_capture(published);

    {
        const std::lock_guard<std::mutex> guard(mutex_);
        if (published) {
            ++stats_.spills_published;
            stats_.spilled_bytes += bytes;
        } else {
            ++stats_.spills_dropped;
        }
    }

    try {
        const std::lock_guard<std::mutex> io(io_mutex_);
        store_.enforce_capacity();
        if (store_.compaction_due()) { (void)store_.compact(cancel_); }
    } catch (const std::exception&) {
        // Capacity maintenance is best effort; the published record stays valid either way.
    }
}

std::optional<DiskRecordDescriptor>
ContextDiskTier::lookup(std::span<const DiskRecordKey> candidates) {
    std::optional<DiskRecordDescriptor> found = store_.lookup_longest(candidates);
    const std::lock_guard<std::mutex> guard(mutex_);
    ++stats_.lookups;
    if (found) { ++stats_.hits; }
    return found;
}

bool ContextDiskTier::restore(const DiskRecordDescriptor& record) {
    const std::optional<DiskCheckpointGeometry> geometry = port_.begin_restore(record);
    if (!geometry || geometry->page_bytes != record.page_bytes ||
        geometry->page_count != record.page_count) {
        if (geometry) { port_.end_restore(false); }
        const std::lock_guard<std::mutex> guard(mutex_);
        ++stats_.restore_failures;
        return false;
    }

    bool complete       = true;
    std::uint64_t bytes = 0;
    try {
        const std::lock_guard<std::mutex> io(io_mutex_);
        if (record.state_bytes != 0) {
            const std::span<std::byte> staging = port_.restore_staging(1);
            if (staging.size() < record.state_bytes) { throw std::runtime_error("short staging"); }
            store_.read_state_image(record, staging.first(record.state_bytes));
            if (!port_.commit_state_image(staging.first(record.state_bytes))) {
                throw std::runtime_error("state image commit failed");
            }
            bytes += record.state_bytes;
        }
        const std::uint32_t batch = std::max(1U, port_.batch_pages());
        for (std::uint32_t first = 0; first < record.page_count; first += batch) {
            const std::uint32_t count          = std::min(batch, record.page_count - first);
            const std::span<std::byte> staging = port_.restore_staging(count);
            const std::size_t needed = static_cast<std::size_t>(count) * record.page_bytes;
            if (staging.size() < needed) { throw std::runtime_error("short staging"); }
            for (std::uint32_t page = 0; page < count; ++page) {
                store_.read_page(record, first + page,
                                 staging.subspan(static_cast<std::size_t>(page) * record.page_bytes,
                                                 record.page_bytes));
            }
            if (!port_.commit_pages(first, count)) {
                throw std::runtime_error("page commit failed");
            }
            bytes += needed;
        }
    } catch (const std::exception&) {
        complete = false;
    }
    port_.end_restore(complete);

    const std::lock_guard<std::mutex> guard(mutex_);
    if (complete) {
        ++stats_.restores;
        stats_.restored_bytes += bytes;
    } else {
        ++stats_.restore_failures;
    }
    return complete;
}

void ContextDiskTier::drain() {
    std::unique_lock<std::mutex> guard(mutex_);
    idle_.wait(guard, [this] { return !running_ || (queue_.empty() && !busy_); });
}

ContextDiskTierStats ContextDiskTier::stats() const {
    ContextDiskStoreStats store = store_.stats();
    const std::lock_guard<std::mutex> guard(mutex_);
    ContextDiskTierStats snapshot = stats_;
    snapshot.store                = store;
    return snapshot;
}

} // namespace ninfer::runtime
