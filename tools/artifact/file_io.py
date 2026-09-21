"""Keep large offline transfers from retaining whole artifacts in Linux's page cache."""

from __future__ import annotations

import mmap
import os

IO_CHUNK_BYTES = 8 * 1024 * 1024
WRITEBACK_BYTES = 64 * 1024 * 1024

# The Windows CRT opens descriptors in text mode by default, which translates CRLF and stops a
# read at the first 0x1A. Every os.open below carries this flag; it is 0 where it does not exist.
O_BINARY = getattr(os, "O_BINARY", 0)

# posix_fadvise and sysconf are Linux-only. Elsewhere the cache hints below are dropped and the
# platform manages its own page cache; conversion output is unaffected either way.
_HAVE_FADVISE = hasattr(os, "posix_fadvise")
_PAGE_BYTES = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else mmap.PAGESIZE
_fdatasync = getattr(os, "fdatasync", os.fsync)


if hasattr(os, "pread"):
    pwrite = os.pwrite

    def pread(fd: int, count: int, offset: int) -> bytes:
        return os.pread(fd, count, offset)

else:
    # Windows has no positional read/write. No descriptor here is shared across threads, so
    # seeking first is equivalent. Both loop, because a single transfer may be short.
    def pread(fd: int, count: int, offset: int) -> bytes:
        chunks: list[bytes] = []
        remaining = count
        os.lseek(fd, offset, os.SEEK_SET)
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def pwrite(fd: int, data, offset: int) -> int:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.write(fd, data)


def discard_cached_pages(fd: int, offset: int = 0, count: int | None = None) -> None:
    if not _HAVE_FADVISE:
        return
    if count is None:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    elif count > 0:
        begin = offset // _PAGE_BYTES * _PAGE_BYTES
        end = (offset + count + _PAGE_BYTES - 1) // _PAGE_BYTES * _PAGE_BYTES
        os.posix_fadvise(fd, begin, end - begin, os.POSIX_FADV_DONTNEED)


class Writeback:
    """Bound dirty output across all open shards; release clean pages after writeback."""

    def __init__(self) -> None:
        self._bytes = 0
        self._fds: set[int] = set()

    def written(self, fd: int, count: int) -> None:
        self._fds.add(fd)
        self._bytes += count
        if self._bytes >= WRITEBACK_BYTES:
            self.flush()

    def flush(self) -> None:
        for fd in self._fds:
            _fdatasync(fd)
            discard_cached_pages(fd)
        self._fds.clear()
        self._bytes = 0
