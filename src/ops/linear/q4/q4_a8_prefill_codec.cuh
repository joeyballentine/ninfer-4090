#pragma once

// Q4 G64 row-split codes -> unscaled E4M3 bytes.
//
// A base byte holds two lanes: the even lane in the low nibble, the odd lane in the high nibble
// (storage-layouts.md 3.3). A nibble is a 4-bit two's-complement word, so lane value
// v = n < 8 ? n : n - 16 lies in [-8,7]. Every one of those sixteen integers is exactly
// representable in E4M3 (three mantissa bits carry integers up to 16 exactly, and 8 is far below
// the 448 finite maximum), so the table below is an exact recode, not a quantization. The group
// scale is deliberately left out: the GEMM folds it into FP32 after the slab's MMA steps.

#include <cstdint>

namespace ninfer::ops::detail {

// Byte i of kQ4NonNegative is E4M3(i); byte i of kQ4Negative is E4M3(i - 8).
//   |v| : 1->0x38 2->0x40 3->0x44 4->0x48 5->0x4A 6->0x4C 7->0x4E 8->0x50, sign bit 0x80.
inline constexpr unsigned long long kQ4E4m3NonNegative = 0x4E4C4A4844403800ULL;
inline constexpr unsigned long long kQ4E4m3Negative    = 0xB8C0C4C8CACCCED0ULL;

__device__ __forceinline__ unsigned e4m3_byte_from_int4(unsigned code) {
    const unsigned long long table = (code & 8U) != 0U ? kQ4E4m3Negative : kQ4E4m3NonNegative;
    return static_cast<unsigned>((table >> (8U * (code & 7U))) & 0xffULL);
}

struct Q4A8PrefillCodec {
    static constexpr int kHighBytesPerGroup = 0;

    // Word `word` covers logical K values 4*word .. 4*word+3 of one staged 32-byte group, which
    // are exactly base bytes 2*word and 2*word+1.
    __device__ static __forceinline__ unsigned
    decode_word(const std::uint8_t* codes, const std::uint8_t* /*high*/, int word) {
        const unsigned packed = *reinterpret_cast<const std::uint16_t*>(codes + 2 * word);
        return e4m3_byte_from_int4(packed & 0xfU) |
               (e4m3_byte_from_int4((packed >> 4) & 0xfU) << 8) |
               (e4m3_byte_from_int4((packed >> 8) & 0xfU) << 16) |
               (e4m3_byte_from_int4((packed >> 12) & 0xfU) << 24);
    }
};

} // namespace ninfer::ops::detail
