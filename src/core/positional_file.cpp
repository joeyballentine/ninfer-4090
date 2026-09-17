#include "core/positional_file.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace ninfer::core {
namespace {

#if defined(_WIN32)

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       DWORD error = ::GetLastError()) {
    throw FileIoError(path.string() + ": " + operation + ": Win32 error " +
                      std::to_string(static_cast<unsigned long>(error)));
}

constexpr DWORD kChunk = 64UL * 1024UL * 1024UL;

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw FileIoError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(const std::filesystem::path& path, std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw FileIoError(path.string() + ": file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

constexpr std::size_t kChunk = 64ULL * 1024ULL * 1024ULL;

#endif

} // namespace

AlignedBuffer::AlignedBuffer(std::size_t bytes) { resize(bytes); }

AlignedBuffer::~AlignedBuffer() { reset(); }

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        data_  = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}

void AlignedBuffer::reset() noexcept {
    if (data_ != nullptr) {
#if defined(_WIN32)
        ::_aligned_free(data_);
#else
        std::free(data_);
#endif
    }
    data_  = nullptr;
    bytes_ = 0;
}

void AlignedBuffer::resize(std::size_t bytes) {
    reset();
    if (bytes == 0) { return; }
    const std::size_t rounded =
        static_cast<std::size_t>(align_up_direct(static_cast<std::uint64_t>(bytes)));
#if defined(_WIN32)
    void* memory = ::_aligned_malloc(rounded, static_cast<std::size_t>(kDirectIoAlignment));
    if (memory == nullptr) { throw std::bad_alloc(); }
#else
    void* memory = nullptr;
    if (::posix_memalign(&memory, static_cast<std::size_t>(kDirectIoAlignment), rounded) != 0) {
        throw std::bad_alloc();
    }
#endif
    data_  = static_cast<std::byte*>(memory);
    bytes_ = bytes;
}

bool PositionalFile::direct_eligible(std::uint64_t offset, const void* buffer,
                                     std::size_t bytes) noexcept {
    return bytes != 0 && offset % kDirectIoAlignment == 0 && bytes % kDirectIoAlignment == 0 &&
           reinterpret_cast<std::uintptr_t>(buffer) % kDirectIoAlignment == 0;
}

void PositionalFile::require_open() const {
    if (!is_open()) { throw FileIoError(path_.string() + ": file is not open"); }
}

#if defined(_WIN32)

PositionalFile::PositionalFile(std::filesystem::path path, FileMode mode, FileAccess access)
    : path_(std::move(path)), access_(access) {
    const DWORD desired =
        access == FileAccess::ReadWrite ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    const DWORD disposition = mode == FileMode::CreateOrOpen ? OPEN_ALWAYS : OPEN_EXISTING;
    handle_ = ::CreateFileW(path_.c_str(), desired, FILE_SHARE_READ, nullptr, disposition,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
        handle_ = nullptr;
        fail(path_, "CreateFileW");
    }
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(handle_, &size) || size.QuadPart < 0) {
        const auto error = ::GetLastError();
        ::CloseHandle(handle_);
        handle_ = nullptr;
        fail(path_, "GetFileSizeEx", error);
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
}

bool PositionalFile::is_open() const noexcept { return handle_ != nullptr; }

void PositionalFile::close() noexcept {
    if (direct_handle_ != nullptr) { ::CloseHandle(direct_handle_); }
    if (handle_ != nullptr) { ::CloseHandle(handle_); }
    direct_handle_ = nullptr;
    handle_        = nullptr;
}

void PositionalFile::open_direct() const {
    if (direct_handle_ != nullptr || direct_unavailable_) { return; }
    const DWORD desired =
        access_ == FileAccess::ReadWrite ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    void* opened = ::CreateFileW(path_.c_str(), desired, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                     FILE_FLAG_WRITE_THROUGH | FILE_FLAG_OVERLAPPED,
                                 nullptr);
    if (opened == INVALID_HANDLE_VALUE) {
        direct_unavailable_ = true;
        return;
    }
    direct_handle_ = opened;
}

namespace {

DWORD overlapped_transfer(void* handle, const std::filesystem::path& path, std::uint64_t offset,
                          void* buffer, DWORD count, bool write, const char* operation) {
    OVERLAPPED state{};
    state.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    state.OffsetHigh = static_cast<DWORD>(offset >> 32U);
    DWORD moved      = 0;
    const BOOL ok    = write ? ::WriteFile(handle, buffer, count, &moved, &state)
                             : ::ReadFile(handle, buffer, count, &moved, &state);
    if (!ok) {
        const auto error = ::GetLastError();
        if (!write && error == ERROR_HANDLE_EOF) { return 0; }
        if (error != ERROR_IO_PENDING) { fail(path, operation, error); }
        if (!::GetOverlappedResult(handle, &state, &moved, TRUE)) {
            const auto wait_error = ::GetLastError();
            if (write || wait_error != ERROR_HANDLE_EOF) { fail(path, operation, wait_error); }
            return 0;
        }
    }
    return moved;
}

} // namespace

std::size_t PositionalFile::read_at(std::uint64_t offset, std::span<std::byte> destination) const {
    require_open();
    if (destination.empty()) { return 0; }
    if (direct_eligible(offset, destination.data(), destination.size())) {
        open_direct();
        if (direct_handle_ != nullptr) {
            std::size_t total = 0;
            while (total < destination.size()) {
                const auto count =
                    static_cast<DWORD>(std::min<std::size_t>(destination.size() - total, kChunk));
                const DWORD moved =
                    overlapped_transfer(direct_handle_, path_, offset + total,
                                        destination.data() + total, count, false, "direct read");
                total += moved;
                if (moved != count) { break; }
            }
            ++direct_transfers_;
            return total;
        }
    }
    std::size_t total = 0;
    while (total < destination.size()) {
        const auto count =
            static_cast<DWORD>(std::min<std::size_t>(destination.size() - total, kChunk));
        const DWORD moved = overlapped_transfer(handle_, path_, offset + total,
                                                destination.data() + total, count, false, "read");
        total += moved;
        if (moved != count) { break; }
    }
    return total;
}

void PositionalFile::write_at(std::uint64_t offset, std::span<const std::byte> source) {
    require_open();
    if (access_ != FileAccess::ReadWrite) {
        throw FileIoError(path_.string() + ": file is not writable");
    }
    if (source.empty()) { return; }
    void* const mutable_source = const_cast<std::byte*>(source.data());
    if (direct_eligible(offset, source.data(), source.size())) {
        open_direct();
        if (direct_handle_ != nullptr) {
            std::size_t total = 0;
            while (total < source.size()) {
                const auto count =
                    static_cast<DWORD>(std::min<std::size_t>(source.size() - total, kChunk));
                const DWORD moved = overlapped_transfer(
                    direct_handle_, path_, offset + total,
                    static_cast<std::byte*>(mutable_source) + total, count, true, "direct write");
                if (moved != count) { fail(path_, "short direct write", ERROR_WRITE_FAULT); }
                total += moved;
            }
            ++direct_transfers_;
            bytes_ = std::max(bytes_, offset + source.size());
            return;
        }
    }
    std::size_t total = 0;
    while (total < source.size()) {
        const auto count = static_cast<DWORD>(std::min<std::size_t>(source.size() - total, kChunk));
        const DWORD moved =
            overlapped_transfer(handle_, path_, offset + total,
                                static_cast<std::byte*>(mutable_source) + total, count, true,
                                "write");
        if (moved != count) { fail(path_, "short write", ERROR_WRITE_FAULT); }
        total += moved;
    }
    bytes_ = std::max(bytes_, offset + source.size());
}

void PositionalFile::truncate(std::uint64_t bytes) {
    require_open();
    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(bytes);
    if (!::SetFilePointerEx(handle_, position, nullptr, FILE_BEGIN)) {
        fail(path_, "SetFilePointerEx");
    }
    if (!::SetEndOfFile(handle_)) { fail(path_, "SetEndOfFile"); }
    bytes_ = bytes;
}

void PositionalFile::flush() {
    require_open();
    if (access_ != FileAccess::ReadWrite) { return; }
    if (!::FlushFileBuffers(handle_)) { fail(path_, "FlushFileBuffers"); }
}

void sync_directory(const std::filesystem::path&) {
    // NTFS commits the directory entry with the file's own flush; there is no directory handle
    // to sync separately.
}

#else

PositionalFile::PositionalFile(std::filesystem::path path, FileMode mode, FileAccess access)
    : path_(std::move(path)), access_(access) {
    int flags = access == FileAccess::ReadWrite ? O_RDWR : O_RDONLY;
    flags |= O_CLOEXEC;
    if (mode == FileMode::CreateOrOpen) { flags |= O_CREAT; }
    fd_ = ::open(path_.c_str(), flags, 0644);
    if (fd_ < 0) { fail(path_, "open"); }

    struct stat status {};
    if (::fstat(fd_, &status) != 0) {
        const auto error = errno;
        ::close(fd_);
        fd_   = -1;
        errno = error;
        fail(path_, "fstat");
    }
    if (status.st_size < 0 || !S_ISREG(status.st_mode)) {
        ::close(fd_);
        fd_ = -1;
        throw FileIoError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

bool PositionalFile::is_open() const noexcept { return fd_ >= 0; }

void PositionalFile::close() noexcept {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
    direct_fd_ = -1;
    fd_        = -1;
}

void PositionalFile::open_direct() const {
#if defined(O_DIRECT)
    if (direct_fd_ >= 0 || direct_unavailable_) { return; }
    const int flags = (access_ == FileAccess::ReadWrite ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_DIRECT;
    const int opened = ::open(path_.c_str(), flags);
    // Filesystems such as tmpfs reject O_DIRECT outright. The buffered path stays correct, so a
    // rejection is remembered and never retried rather than reported as an error.
    if (opened < 0) {
        direct_unavailable_ = true;
        return;
    }
    direct_fd_ = opened;
#else
    direct_unavailable_ = true;
#endif
}

std::size_t PositionalFile::read_at(std::uint64_t offset, std::span<std::byte> destination) const {
    require_open();
    if (destination.empty()) { return 0; }
    if (direct_eligible(offset, destination.data(), destination.size())) {
        open_direct();
        if (direct_fd_ >= 0) {
            std::size_t total = 0;
            while (total < destination.size()) {
                const std::size_t count = std::min(destination.size() - total, kChunk);
                const ssize_t moved     = ::pread(direct_fd_, destination.data() + total, count,
                                                  file_offset(path_, offset + total));
                if (moved < 0) {
                    if (errno == EINTR) { continue; }
                    fail(path_, "direct pread");
                }
                total += static_cast<std::size_t>(moved);
                if (static_cast<std::size_t>(moved) != count) { break; }
            }
            ++direct_transfers_;
            return total;
        }
    }
    std::size_t total = 0;
    while (total < destination.size()) {
        const std::size_t count = std::min(destination.size() - total, kChunk);
        const ssize_t moved =
            ::pread(fd_, destination.data() + total, count, file_offset(path_, offset + total));
        if (moved < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        total += static_cast<std::size_t>(moved);
        if (static_cast<std::size_t>(moved) != count) { break; }
    }
    return total;
}

void PositionalFile::write_at(std::uint64_t offset, std::span<const std::byte> source) {
    require_open();
    if (access_ != FileAccess::ReadWrite) {
        throw FileIoError(path_.string() + ": file is not writable");
    }
    if (source.empty()) { return; }
    const bool direct = direct_eligible(offset, source.data(), source.size());
    if (direct) { open_direct(); }
    const int target = direct && direct_fd_ >= 0 ? direct_fd_ : fd_;
    if (target == direct_fd_ && direct) { ++direct_transfers_; }
    std::size_t total = 0;
    while (total < source.size()) {
        const std::size_t count = std::min(source.size() - total, kChunk);
        const ssize_t moved =
            ::pwrite(target, source.data() + total, count, file_offset(path_, offset + total));
        if (moved < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pwrite");
        }
        if (moved == 0) { throw FileIoError(path_.string() + ": write made no progress"); }
        total += static_cast<std::size_t>(moved);
    }
    bytes_ = std::max(bytes_, offset + source.size());
}

void PositionalFile::truncate(std::uint64_t bytes) {
    require_open();
    if (::ftruncate(fd_, file_offset(path_, bytes)) != 0) { fail(path_, "ftruncate"); }
    bytes_ = bytes;
}

void PositionalFile::flush() {
    require_open();
    if (access_ != FileAccess::ReadWrite) { return; }
    if (direct_fd_ >= 0 && ::fsync(direct_fd_) != 0) { fail(path_, "fsync direct"); }
    if (::fsync(fd_) != 0) { fail(path_, "fsync"); }
}

void sync_directory(const std::filesystem::path& directory) {
    const int fd = ::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    if (fd < 0) { fail(directory, "open directory"); }
    const int result = ::fsync(fd);
    const int error  = errno;
    ::close(fd);
    // Some filesystems reject fsync on a directory handle; the rename is already ordered there.
    if (result != 0 && error != EINVAL) {
        errno = error;
        fail(directory, "fsync directory");
    }
}

#endif

PositionalFile::~PositionalFile() { close(); }

PositionalFile::PositionalFile(PositionalFile&& other) noexcept
    : path_(std::move(other.path_)), access_(other.access_),
#if defined(_WIN32)
      handle_(std::exchange(other.handle_, nullptr)),
      direct_handle_(std::exchange(other.direct_handle_, nullptr)),
#else
      fd_(std::exchange(other.fd_, -1)), direct_fd_(std::exchange(other.direct_fd_, -1)),
#endif
      bytes_(std::exchange(other.bytes_, 0)),
      direct_transfers_(std::exchange(other.direct_transfers_, 0)),
      direct_unavailable_(other.direct_unavailable_) {
}

PositionalFile& PositionalFile::operator=(PositionalFile&& other) noexcept {
    if (this != &other) {
        close();
        path_   = std::move(other.path_);
        access_ = other.access_;
#if defined(_WIN32)
        handle_        = std::exchange(other.handle_, nullptr);
        direct_handle_ = std::exchange(other.direct_handle_, nullptr);
#else
        fd_        = std::exchange(other.fd_, -1);
        direct_fd_ = std::exchange(other.direct_fd_, -1);
#endif
        bytes_              = std::exchange(other.bytes_, 0);
        direct_transfers_   = std::exchange(other.direct_transfers_, 0);
        direct_unavailable_ = other.direct_unavailable_;
    }
    return *this;
}

void PositionalFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (read_at(offset, destination) != destination.size()) {
        throw FileIoError(path_.string() + ": unexpected end of file");
    }
}

std::uint64_t PositionalFile::append(std::span<const std::byte> source) {
    const std::uint64_t offset = bytes_;
    write_at(offset, source);
    return offset;
}

void atomic_replace_file(const std::filesystem::path& source,
                         const std::filesystem::path& destination) {
    std::error_code code;
    std::filesystem::rename(source, destination, code);
    if (code) {
        throw FileIoError(destination.string() + ": rename from " + source.string() + " failed: " +
                          code.message());
    }
    sync_directory(destination.parent_path());
}

} // namespace ninfer::core
