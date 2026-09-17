#pragma once

// Q5 G64 row-split codes -> unscaled E4M3 bytes.
//
// The base plane is the same nibble pair as Q4; bit 4 of each lane lives in the lane-major
// high-bit plane, eight bytes per group (storage-layouts.md 3.3/3.4). The assembled 5-bit word is
// two's complement, so v = c < 16 ? c : c - 32 lies in [-16,15]. Every one of those integers is
// exactly representable in E4M3 (|v| <= 16 needs at most three mantissa bits), so this is an exact
// recode. The group scale is applied by the GEMM in FP32, not here.

#include <cstdint>

namespace ninfer::ops::detail {

// Byte i of each table is E4M3 of the value that 5-bit code (8*table_index + i) denotes.
//   |v| : 1->0x38 2->0x40 3->0x44 4->0x48 5->0x4A 6->0x4C 7->0x4E
//         8->0x50 9->0x51 10->0x52 11->0x53 12->0x54 13->0x55 14->0x56 15->0x57 16->0x58,
//   sign bit 0x80.
inline constexpr unsigned long long kQ5E4m3Code0 = 0x4E4C4A4844403800ULL;  // codes  0..7  -> 0..7
inline constexpr unsigned long long kQ5E4m3Code8 = 0x5756555453525150ULL;  // codes  8..15 -> 8..15
inline constexpr unsigned long long kQ5E4m3Code16 = 0xD1D2D3D4D5D6D7D8ULL; // codes 16..23 -> -16..-9
inline constexpr unsigned long long kQ5E4m3Code24 = 0xB8C0C4C8CACCCED0ULL; // codes 24..31 -> -8..-1

__device__ __forceinline__ unsigned e4m3_byte_from_int5(unsigned code) {
    const unsigned long long low  = (code & 8U) != 0U ? kQ5E4m3Code8 : kQ5E4m3Code0;
    const unsigned long long high = (code & 8U) != 0U ? kQ5E4m3Code24 : kQ5E4m3Code16;
    const unsigned long long table = (code & 16U) != 0U ? high : low;
    return static_cast<unsigned>((table >> (8U * (code & 7U))) & 0xffULL);
}

struct Q5A8PrefillCodec {
    static constexpr int kHighBytesPerGroup = 8;

    // Word `word` covers logical K values 4*word .. 4*word+3: base bytes 2*word and 2*word+1, and
    // four adjacent bits of high byte word/2 (the low nibble for even words, the high nibble for
    // odd ones), because stream bit t sits in bit t % 8 of high byte t / 8.
    __device__ static __forceinline__ unsigned
    decode_word(const std::uint8_t* codes, const std::uint8_t* high, int word) {
        const unsigned packed = *reinterpret_cast<const std::uint16_t*>(codes + 2 * word);
        const unsigned bits   = (high[word >> 1] >> (4 * (word & 1))) & 0xfU;
        return e4m3_byte_from_int5((packed & 0xfU) | ((bits & 1U) << 4)) |
               (e4m3_byte_from_int5(((packed >> 4) & 0xfU) | (((bits >> 1) & 1U) << 4)) << 8) |
               (e4m3_byte_from_int5(((packed >> 8) & 0xfU) | (((bits >> 2) & 1U) << 4)) << 16) |
               (e4m3_byte_from_int5(((packed >> 12) & 0xfU) | (((bits >> 3) & 1U) << 4)) << 24);
    }
};

} // namespace ninfer::ops::detail
