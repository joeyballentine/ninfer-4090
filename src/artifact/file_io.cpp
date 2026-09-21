#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
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

namespace ninfer::artifact {
namespace {

#if defined(_WIN32)

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation,
                       DWORD error = ::GetLastError()) {
    throw ArtifactError(path.string() + ": " + operation + ": Win32 error " +
                        std::to_string(static_cast<unsigned long>(error)));
}

void* open_handle(const std::filesystem::path& path, DWORD flags) {
    // POSIX descriptors place no lock on the file. Sharing only reads would keep an open artifact
    // from being replaced or deleted for as long as the Engine holds it.
    constexpr DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    void* const handle =
        ::CreateFileW(path.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING, flags, nullptr);
    return handle == INVALID_HANDLE_VALUE ? nullptr : handle;
}

// One positional overlapped read. Returns the byte count; a count below `count` means EOF.
DWORD read_at(void* handle, const std::filesystem::path& path, std::uint64_t offset, void* buffer,
              DWORD count, const char* operation) {
    OVERLAPPED operation_state{};
    operation_state.Offset     = static_cast<DWORD>(offset & 0xffffffffULL);
    operation_state.OffsetHigh = static_cast<DWORD>(offset >> 32U);

    DWORD read = 0;
    if (!::ReadFile(handle, buffer, count, &read, &operation_state)) {
        const auto error = ::GetLastError();
        if (error == ERROR_HANDLE_EOF) { return 0; }
        if (error != ERROR_IO_PENDING) { fail(path, operation, error); }
        if (!::GetOverlappedResult(handle, &operation_state, &read, TRUE)) {
            const auto wait_error = ::GetLastError();
            if (wait_error != ERROR_HANDLE_EOF) { fail(path, operation, wait_error); }
            return 0;
        }
    }
    return read;
}

#else

[[noreturn]] void fail(const std::filesystem::path& path, const char* operation) {
    throw ArtifactError(path.string() + ": " + operation + ": " + std::strerror(errno));
}

off_t file_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
        throw ArtifactError("file offset exceeds positional I/O range");
    }
    return static_cast<off_t>(offset);
}

#endif

} // namespace

#if defined(_WIN32)

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    handle_ = open_handle(path_, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED);
    if (handle_ == nullptr) { fail(path_, "CreateFileW"); }

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(handle_, &size)) {
        const auto error = ::GetLastError();
        ::CloseHandle(handle_);
        handle_ = nullptr;
        fail(path_, "GetFileSizeEx", error);
    }
    if (size.QuadPart < 0) {
        ::CloseHandle(handle_);
        handle_ = nullptr;
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(size.QuadPart);
}

InputFile::~InputFile() {
    if (direct_handle_ != nullptr) { ::CloseHandle(direct_handle_); }
    if (handle_ != nullptr) { ::CloseHandle(handle_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count =
            static_cast<DWORD>(std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024));
        const DWORD read = read_at(handle_, path_, offset, destination.data(), count, "ReadFile");
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += read;
        destination = destination.subspan(read);
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_handle_ == nullptr) {
        direct_handle_ = open_handle(path_, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                                                FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN);
        if (direct_handle_ == nullptr) { fail(path_, "CreateFileW direct"); }
    }
    // Unbuffered reads are capped so a single request stays inside the DWORD byte count.
    constexpr std::size_t kMaxDirectRead = 1ULL << 30;
    std::size_t total                    = 0;
    while (total < destination.size()) {
        const auto count = static_cast<DWORD>(std::min(kMaxDirectRead, destination.size() - total));
        const DWORD read = read_at(direct_handle_, path_, offset + total,
                                   destination.data() + total, count, "direct ReadFile");
        total += read;
        if (read != count) { break; }
    }
    return total;
}

#else

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
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
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    bytes_ = static_cast<std::uint64_t>(status.st_size);
}

InputFile::~InputFile() {
    if (direct_fd_ >= 0) { ::close(direct_fd_); }
    if (fd_ >= 0) { ::close(fd_); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    while (!destination.empty()) {
        const auto count = std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024);
        const auto read  = ::pread(fd_, destination.data(), count, file_offset(offset));
        if (read < 0) {
            if (errno == EINTR) { continue; }
            fail(path_, "pread");
        }
        if (!read) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(read);
        destination = destination.subspan(static_cast<std::size_t>(read));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (direct_fd_ < 0) {
        direct_fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (direct_fd_ < 0) { fail(path_, "open direct"); }
    }
    ssize_t read;
    do {
        read = ::pread(direct_fd_, destination.data(), destination.size(), file_offset(offset));
    } while (read < 0 && errno == EINTR);
    if (read < 0) { fail(path_, "direct pread"); }
    return static_cast<std::size_t>(read);
}

#endif

} // namespace ninfer::artifact
