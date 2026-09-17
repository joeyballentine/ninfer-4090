#pragma once

// Positional read/write file primitive for persistent context storage.
//
// Core owns raw transfers, so this is the only place that talks to the operating system's
// positional I/O. It is deliberately narrower than `artifact::InputFile`: that type is read-only
// and lives above Core, so it cannot serve a store that also appends and flushes. The direct
// (unbuffered) technique is the same one - `pread`/`pwrite` against a second `O_DIRECT`
// descriptor on POSIX, a second `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` handle on
// Windows - and applies only when offset, length and buffer address are all sector aligned.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::core {

// Sector size assumed by every direct transfer. Both supported platforms accept 4096 for the
// storage classes this engine targets; a smaller physical sector is always a divisor of it.
inline constexpr std::uint64_t kDirectIoAlignment = 4096;

[[nodiscard]] constexpr std::uint64_t align_up_direct(std::uint64_t bytes) noexcept {
    return (bytes + kDirectIoAlignment - 1U) & ~(kDirectIoAlignment - 1U);
}

class FileIoError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Page-aligned host buffer. Direct transfers require the buffer address to be sector aligned,
// which neither `new` nor `std::vector` guarantees.
class AlignedBuffer {
public:
    AlignedBuffer() noexcept = default;
    explicit AlignedBuffer(std::size_t bytes);
    ~AlignedBuffer();
    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    void resize(std::size_t bytes);
    void reset() noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return bytes_; }
    [[nodiscard]] std::byte* data() noexcept { return data_; }
    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::span<std::byte> span() noexcept { return {data_, bytes_}; }
    [[nodiscard]] std::span<const std::byte> span() const noexcept { return {data_, bytes_}; }

private:
    std::byte* data_   = nullptr;
    std::size_t bytes_ = 0;
};

enum class FileMode : std::uint8_t {
    OpenExisting, // Fails when the file is absent.
    CreateOrOpen, // Creates an empty file when absent; never truncates an existing one.
};

enum class FileAccess : std::uint8_t {
    Read,
    ReadWrite,
};

class PositionalFile {
public:
    PositionalFile() noexcept = default;
    PositionalFile(std::filesystem::path path, FileMode mode, FileAccess access);
    ~PositionalFile();
    PositionalFile(const PositionalFile&)            = delete;
    PositionalFile& operator=(const PositionalFile&) = delete;
    PositionalFile(PositionalFile&& other) noexcept;
    PositionalFile& operator=(PositionalFile&& other) noexcept;

    [[nodiscard]] bool is_open() const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

    // Reads up to `destination.size()`. A shorter result means end of file.
    [[nodiscard]] std::size_t read_at(std::uint64_t offset, std::span<std::byte> destination) const;
    // Reads the full range or throws.
    void read_exact(std::uint64_t offset, std::span<std::byte> destination) const;

    void write_at(std::uint64_t offset, std::span<const std::byte> source);
    // Writes at the current end of file and returns the offset the bytes landed at.
    std::uint64_t append(std::span<const std::byte> source);

    void truncate(std::uint64_t bytes);
    // Returns only once the written bytes and the file length are durable on the device.
    void flush();
    void close() noexcept;

    // True when a transfer of this geometry can use the unbuffered descriptor.
    [[nodiscard]] static bool direct_eligible(std::uint64_t offset, const void* buffer,
                                              std::size_t bytes) noexcept;
    // Direct transfers actually performed by this handle; used by tests and diagnostics.
    [[nodiscard]] std::uint64_t direct_transfers() const noexcept { return direct_transfers_; }

private:
    void require_open() const;
    void open_direct() const;

    std::filesystem::path path_;
    FileAccess access_ = FileAccess::Read;
#if defined(_WIN32)
    void* handle_                = nullptr;
    mutable void* direct_handle_ = nullptr;
#else
    int fd_                = -1;
    mutable int direct_fd_ = -1;
#endif
    std::uint64_t bytes_                      = 0;
    mutable std::uint64_t direct_transfers_   = 0;
    mutable bool direct_unavailable_          = false;
};

// Durably replaces `destination` with `source`. On POSIX the containing directory is synced so
// the rename itself survives a power loss, which is what makes a publish atomic.
void atomic_replace_file(const std::filesystem::path& source,
                         const std::filesystem::path& destination);

// Flushes the directory entry so a create or unlink in it is durable.
void sync_directory(const std::filesystem::path& directory);

} // namespace ninfer::core
