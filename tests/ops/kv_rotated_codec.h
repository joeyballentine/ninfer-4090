#pragma once

// Host replicas of the sm_89 rotated KV codecs (rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8).
//
// Every routine here mirrors, operation for operation, the device code it names so the KV tests
// can assert an exact codec oracle instead of a tolerance:
//   * hadamard64                -> ops::kv_cache_hadamard64            (int8_g64_codec.cuh)
//   * quantize_int8 / int4      -> ops::kv_cache_int8_quant_code / kv_cache_i4_quant_code
//   * e8_project_8d             -> ops::e8_project_8d_warp             (e8_lattice.cuh)
//   * encode_cylinder_8d        -> ops::e8_encode_cylinder_8d_warp     (e8_root_codec.cuh)
//   * decode_cylinder_8d        -> ops::e8_root_decode_8d_fast         (e8_root_codec.cuh)
// The warp-cooperative kernels reduce inside an 8-lane subgroup; the replicas keep the same
// butterfly order and the same tie-breaks so the float results are bit-identical, except where
// `CylinderCode::ambiguous` says the device decision sits on a rounding boundary (see below).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace ninfer::test::rkv {

inline constexpr int kHeadDim = 256;
inline constexpr int kGroup   = 64;
inline constexpr int kGroups  = kHeadDim / kGroup;
// One 8-dim E8 sub-vector per 8 dimensions; two bytes (root, radius|axis) encode each.
inline constexpr int kE8Sub          = 8;
inline constexpr int kE8SubsPerGroup = kGroup / kE8Sub;
inline constexpr int kE8KeyBytes     = kHeadDim / 4;
inline constexpr int kPackedBytes    = kHeadDim / 2;

// ---------------------------------------------------------------------------
// FP16 helpers (RNE), matching __float2half_rn / __half2float.
// ---------------------------------------------------------------------------

inline std::uint16_t half_bits_rne(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t exp  = (bits >> 23) & 0xffu;
    std::uint32_t mantissa   = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }
    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    auto rounded_exp            = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31u) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

inline float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    std::uint32_t result         = 0;
    if (exponent == 0) {
        if (mantissa != 0) {
            float value = std::ldexp(static_cast<float>(mantissa), -24);
            std::memcpy(&result, &value, sizeof(result));
            result |= sign;
        } else {
            result = sign;
        }
    } else if (exponent == 31) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &result, sizeof(out));
    return out;
}

// __float2int_rn
inline std::int32_t round_nearest_even(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    const auto lower     = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) { return lower; }
    if (fraction > 0.5f) { return lower + 1; }
    return (lower & 1) == 0 ? lower : lower + 1;
}

// ---------------------------------------------------------------------------
// H64 rotation. Mirrors ops::kv_cache_hadamard64: lane L of a warp holds dims L
// and L + 32 of one 64-dim group; five XOR butterflies over the lane index are
// followed by one cross-half butterfly scaled by 1/8 (= 1/sqrt(64)).
// ---------------------------------------------------------------------------

inline void hadamard64(std::array<float, kGroup>& values) {
    std::array<float, 32> x0{};
    std::array<float, 32> x1{};
    for (int lane = 0; lane < 32; ++lane) {
        x0[static_cast<std::size_t>(lane)] = values[static_cast<std::size_t>(lane)];
        x1[static_cast<std::size_t>(lane)] = values[static_cast<std::size_t>(lane + 32)];
    }
    for (int offset = 1; offset < 32; offset <<= 1) {
        std::array<float, 32> n0 = x0;
        std::array<float, 32> n1 = x1;
        for (int lane = 0; lane < 32; ++lane) {
            const auto self    = static_cast<std::size_t>(lane);
            const auto partner = static_cast<std::size_t>(lane ^ offset);
            const float y0     = x0[partner];
            const float y1     = x1[partner];
            const bool high    = (lane & offset) != 0;
            n0[self]           = high ? y0 - x0[self] : x0[self] + y0;
            n1[self]           = high ? y1 - x1[self] : x1[self] + y1;
        }
        x0 = n0;
        x1 = n1;
    }
    for (int lane = 0; lane < 32; ++lane) {
        const float a                           = x0[static_cast<std::size_t>(lane)];
        const float b                           = x1[static_cast<std::size_t>(lane)];
        values[static_cast<std::size_t>(lane)]      = (a + b) * 0.125f;
        values[static_cast<std::size_t>(lane + 32)] = (a - b) * 0.125f;
    }
}

// ---------------------------------------------------------------------------
// Scalar group quantization (shared by every rotated mode).
// ---------------------------------------------------------------------------

inline float group_absmax(const std::array<float, kGroup>& values) {
    float absmax = 0.0f;
    for (const float value : values) { absmax = std::max(absmax, std::abs(value)); }
    return absmax;
}

// The stored scale is FP16-RNE(absmax / limit); codes use the reciprocal of the represented value.
inline std::uint16_t group_scale_bits(float absmax, float limit) {
    return half_bits_rne(absmax > 0.0f ? absmax / limit : 0.0f);
}

inline std::int8_t quantize(float value, float inverse_scale, int limit) {
    if (inverse_scale == 0.0f) { return 0; }
    const std::int32_t code = std::clamp(round_nearest_even(value * inverse_scale), -limit, limit);
    return static_cast<std::int8_t>(code);
}

inline std::uint8_t pack_int4(std::int8_t low, std::int8_t high) {
    return static_cast<std::uint8_t>((static_cast<unsigned>(low) & 0x0fu) |
                                     ((static_cast<unsigned>(high) & 0x0fu) << 4));
}

inline std::int8_t unpack_int4(std::uint8_t packed, bool high) {
    const unsigned nibble = high ? (packed >> 4) : (packed & 0x0fu);
    return static_cast<std::int8_t>(static_cast<int>(nibble ^ 8u) - 8);
}

// ---------------------------------------------------------------------------
// E8 lattice projection over one 8-dim sub-vector (rk4v4-e8).
// Mirrors ops::e8_project_8d_warp_single for an 8-lane subgroup.
// ---------------------------------------------------------------------------

// `boundary`, when given, is set if the projection sits on a rounding or coset-distance tie that a
// one-ULP host/device difference (the device compiles with -fmad=1) can legitimately flip.
inline void e8_project_8d(std::array<float, kE8Sub>& x, bool* boundary = nullptr) {
    if (boundary != nullptr) {
        for (const float value : x) {
            const float fraction = std::abs(value - std::floor(value) - 0.5f);
            if (fraction < 1e-4f) { *boundary = true; }
        }
    }
    std::array<float, kE8Sub> f{};
    std::array<float, kE8Sub> f_shift{};
    int sum_f     = 0;
    int sum_shift = 0;
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index = static_cast<std::size_t>(i);
        f[index]         = std::rint(x[index]);
        sum_f += static_cast<int>(f[index]);
        const float shifted = x[index] - 0.5f;
        f_shift[index]      = std::rint(shifted);
        sum_shift += static_cast<int>(f_shift[index]);
    }
    int worst       = 0;
    int worst_shift = 0;
    float max_err   = std::abs(x[0] - f[0]);
    float max_err_s = std::abs((x[0] - 0.5f) - f_shift[0]);
    for (int i = 1; i < kE8Sub; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const float err  = std::abs(x[index] - f[index]);
        if (err > max_err) {
            max_err = err;
            worst   = i;
        }
        const float err_s = std::abs((x[index] - 0.5f) - f_shift[index]);
        if (err_s > max_err_s) {
            max_err_s   = err_s;
            worst_shift = i;
        }
    }
    std::array<float, kE8Sub> d8     = f;
    std::array<float, kE8Sub> coset1{};
    for (int i = 0; i < kE8Sub; ++i) {
        coset1[static_cast<std::size_t>(i)] = f_shift[static_cast<std::size_t>(i)] + 0.5f;
    }
    if ((sum_f & 1) != 0) {
        const auto index = static_cast<std::size_t>(worst);
        d8[index] += (x[index] >= f[index]) ? 1.0f : -1.0f;
    }
    if ((sum_shift & 1) != 0) {
        const auto index = static_cast<std::size_t>(worst_shift);
        coset1[index] += ((x[index] - 0.5f) >= f_shift[index]) ? 1.0f : -1.0f;
    }
    float dist_d8     = 0.0f;
    float dist_coset1 = 0.0f;
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index = static_cast<std::size_t>(i);
        const float a    = x[index] - d8[index];
        const float b    = x[index] - coset1[index];
        dist_d8 += a * a;
        dist_coset1 += b * b;
    }
    if (boundary != nullptr) {
        const float span = std::max(dist_d8, dist_coset1);
        if (std::abs(dist_d8 - dist_coset1) <= 1e-5f * std::max(span, 1.0f)) { *boundary = true; }
    }
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index = static_cast<std::size_t>(i);
        x[index]         = (dist_d8 <= dist_coset1) ? d8[index] : coset1[index];
    }
}

// ---------------------------------------------------------------------------
// E8 cylinder codec (rk2v4-e8): 8-bit index into the 240 minimal E8 roots,
// 4-bit log-scale radius and a 4-bit residual hyperoctahedral axis.
// ---------------------------------------------------------------------------

struct CylinderCode {
    std::uint8_t root     = 0;
    std::uint8_t rad_axis = 0;
    // A boundary decision (radius rounding, Type A/B selection, top-2 or residual-axis tie)
    // that a one-ULP difference between the host replica and the device - fused multiply-add
    // contraction, or a libm logf that is not correctly rounded - can legitimately flip.
    bool ambiguous = false;
};

inline constexpr std::array<std::pair<int, int>, 28> kE8Pairs{{{0, 1},
                                                               {0, 2},
                                                               {0, 3},
                                                               {0, 4},
                                                               {0, 5},
                                                               {0, 6},
                                                               {0, 7},
                                                               {1, 2},
                                                               {1, 3},
                                                               {1, 4},
                                                               {1, 5},
                                                               {1, 6},
                                                               {1, 7},
                                                               {2, 3},
                                                               {2, 4},
                                                               {2, 5},
                                                               {2, 6},
                                                               {2, 7},
                                                               {3, 4},
                                                               {3, 5},
                                                               {3, 6},
                                                               {3, 7},
                                                               {4, 5},
                                                               {4, 6},
                                                               {4, 7},
                                                               {5, 6},
                                                               {5, 7},
                                                               {6, 7}}};

// 4-bit log-radius multipliers, copied from ops::c_radius_scale.
inline constexpr std::array<float, 16> kRadiusScale{0.0000f, 0.0992f, 0.1250f, 0.1575f,
                                                    0.1984f, 0.2500f, 0.3150f, 0.3969f,
                                                    0.5000f, 0.6300f, 0.7937f, 1.0000f,
                                                    1.2599f, 1.5874f, 2.0000f, 2.5198f};

// Generates one entry of ops::c_e8_stage1_i8x8: the int8 direction of root `code` in units of
// 1/4. Codes 0..111 are the 112 Type-A roots (+/-1 in two coordinates), 112..239 the 128 Type-B
// roots (+/-1/2 in all eight, even number of minus signs); 240..255 are unused and decode to 0.
inline std::array<std::int8_t, kE8Sub> e8_root_direction(std::uint8_t code) {
    std::array<std::int8_t, kE8Sub> out{};
    if (code < 112) {
        const auto& pair    = kE8Pairs[static_cast<std::size_t>(code >> 2)];
        const int sign_bits = code & 3;
        out[static_cast<std::size_t>(pair.first)]  = static_cast<std::int8_t>((sign_bits & 2) ? 4 : -4);
        out[static_cast<std::size_t>(pair.second)] = static_cast<std::int8_t>((sign_bits & 1) ? 4 : -4);
    } else if (code < 240) {
        const int b_code = code - 112;
        int parity       = 0;
        for (int i = 0; i < 7; ++i) {
            const int bit = (b_code >> i) & 1;
            parity ^= (1 - bit);
            out[static_cast<std::size_t>(i)] = static_cast<std::int8_t>(bit ? 2 : -2);
        }
        out[7] = static_cast<std::int8_t>(parity == 0 ? 2 : -2);
    }
    return out;
}

// Mirrors ops::e8_root_decode_8d_fast: saturating byte add of the root and axis directions,
// scaled by the 4-bit radius multiplier and rounded to nearest even.
inline std::array<std::int8_t, kE8Sub> decode_cylinder_8d(const CylinderCode& code) {
    std::array<std::int8_t, kE8Sub> out{};
    const unsigned rad_idx  = code.rad_axis >> 4;
    const unsigned axis_idx = code.rad_axis & 0x0fu;
    if (rad_idx == 0) { return out; }
    const auto root  = e8_root_direction(code.root);
    const float mult = kRadiusScale[rad_idx];
    for (int i = 0; i < kE8Sub; ++i) {
        int value = root[static_cast<std::size_t>(i)];
        if (static_cast<unsigned>(i) == (axis_idx >> 1)) { value += (axis_idx & 1u) ? -1 : 1; }
        value = std::clamp(value, -128, 127);
        out[static_cast<std::size_t>(i)] =
            static_cast<std::int8_t>(round_nearest_even(static_cast<float>(value) * mult));
    }
    return out;
}

inline CylinderCode encode_cylinder_8d(const std::array<float, kE8Sub>& value, float ks) {
    CylinderCode result;
    std::array<float, kE8Sub> norm_sq{};
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index = static_cast<std::size_t>(i);
        norm_sq[index]   = value[index] * value[index];
    }
    const auto butterfly_add = [](std::array<float, kE8Sub>& lanes) {
        for (int mask = 4; mask > 0; mask >>= 1) {
            std::array<float, kE8Sub> next = lanes;
            for (int i = 0; i < kE8Sub; ++i) {
                next[static_cast<std::size_t>(i)] =
                    lanes[static_cast<std::size_t>(i)] + lanes[static_cast<std::size_t>(i ^ mask)];
            }
            lanes = next;
        }
    };
    butterfly_add(norm_sq);
    const float out_norm = std::sqrt(norm_sq[0]);
    const float r_rel    = out_norm / (ks * 2.82842712474619f + 1e-8f);
    unsigned rad_idx     = 0;
    if (std::abs(r_rel - 0.08f) < 1e-6f) { result.ambiguous = true; }
    if (r_rel >= 0.08f) {
        const float log_val = 3.0f * (std::log(r_rel) * 1.4426950408889634f) + 8.0f;
        const int q_rad     = static_cast<int>(std::rint(log_val));
        rad_idx             = static_cast<unsigned>(std::clamp(q_rad, 1, 15));
        if (std::abs(log_val - (std::floor(log_val) + 0.5f)) < 1e-3f) { result.ambiguous = true; }
    }

    const float inv_norm = 1.0f / (out_norm + 1e-8f);
    std::array<float, kE8Sub> u{};
    std::array<float, kE8Sub> abs_u{};
    std::array<int, kE8Sub> sign_u{};
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index = static_cast<std::size_t>(i);
        u[index]         = value[index] * inv_norm;
        abs_u[index]     = std::abs(u[index]);
        sign_u[index]    = u[index] >= 0.0f ? 1 : -1;
    }

    // Type A: top-2 |u| over the subgroup, ties resolved toward the lower dimension.
    std::array<float, kE8Sub> top1_val = abs_u;
    std::array<float, kE8Sub> top2_val{};
    std::array<int, kE8Sub> top1_idx{};
    std::array<int, kE8Sub> top2_idx{};
    for (int i = 0; i < kE8Sub; ++i) {
        top1_idx[static_cast<std::size_t>(i)] = i;
        top2_val[static_cast<std::size_t>(i)] = -1.0f;
        top2_idx[static_cast<std::size_t>(i)] = -1;
    }
    for (int mask = 1; mask <= 4; mask <<= 1) {
        auto n1v = top1_val;
        auto n1i = top1_idx;
        auto n2v = top2_val;
        auto n2i = top2_idx;
        for (int i = 0; i < kE8Sub; ++i) {
            const auto self    = static_cast<std::size_t>(i);
            const auto partner = static_cast<std::size_t>(i ^ mask);
            const float o1v    = top1_val[partner];
            const int o1i      = top1_idx[partner];
            const float o2v    = top2_val[partner];
            const int o2i      = top2_idx[partner];
            if (o1v > top1_val[self] || (o1v == top1_val[self] && o1i < top1_idx[self])) {
                n2v[self] = (top1_val[self] > o2v) ? top1_val[self] : o2v;
                n2i[self] = (top1_val[self] > o2v) ? top1_idx[self] : o2i;
                n1v[self] = o1v;
                n1i[self] = o1i;
            } else if (o1v > top2_val[self] || (o1v == top2_val[self] && o1i < top2_idx[self])) {
                n2v[self] = o1v;
                n2i[self] = o1i;
            }
        }
        top1_val = n1v;
        top1_idx = n1i;
        top2_val = n2v;
        top2_idx = n2i;
    }
    const int best_i        = std::min(top1_idx[0], top2_idx[0]);
    const int best_j        = std::max(top1_idx[0], top2_idx[0]);
    const int best_pair_idx = (best_i * (15 - best_i)) / 2 + (best_j - best_i - 1);
    const int s_i_bit       = sign_u[static_cast<std::size_t>(best_i)] > 0 ? 1 : 0;
    const int s_j_bit       = sign_u[static_cast<std::size_t>(best_j)] > 0 ? 1 : 0;
    const auto type_a_code =
        static_cast<std::uint8_t>(best_pair_idx * 4 + (s_i_bit << 1) + s_j_bit);
    const float type_a_score = top1_val[0] + top2_val[0];

    // Type B: half-integer root, one sign per coordinate with an even number of minus signs.
    std::array<float, kE8Sub> sum_abs = abs_u;
    std::array<float, kE8Sub> min_abs = abs_u;
    std::array<int, kE8Sub> min_idx{};
    std::array<int, kE8Sub> minus_count{};
    for (int i = 0; i < kE8Sub; ++i) {
        min_idx[static_cast<std::size_t>(i)]     = i;
        minus_count[static_cast<std::size_t>(i)] = sign_u[static_cast<std::size_t>(i)] < 0 ? 1 : 0;
    }
    for (int mask = 4; mask > 0; mask >>= 1) {
        auto ns = sum_abs;
        auto nm = minus_count;
        auto nv = min_abs;
        auto ni = min_idx;
        for (int i = 0; i < kE8Sub; ++i) {
            const auto self    = static_cast<std::size_t>(i);
            const auto partner = static_cast<std::size_t>(i ^ mask);
            ns[self]           = sum_abs[self] + sum_abs[partner];
            nm[self]           = minus_count[self] + minus_count[partner];
            const float other  = min_abs[partner];
            const int other_id = min_idx[partner];
            if (other < min_abs[self] || (other == min_abs[self] && other_id < min_idx[self])) {
                nv[self] = other;
                ni[self] = other_id;
            }
        }
        sum_abs     = ns;
        minus_count = nm;
        min_abs     = nv;
        min_idx     = ni;
    }
    float type_b_score = 0.5f * sum_abs[0];
    const bool odd     = (minus_count[0] & 1) != 0;
    if (odd) { type_b_score -= min_abs[0]; }
    std::array<int, kE8Sub> b_sign{};
    unsigned b_bit = 0;
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index    = static_cast<std::size_t>(i);
        b_sign[index]       = (i == min_idx[0] && odd) ? -sign_u[index] : sign_u[index];
        if (i < 7 && b_sign[index] > 0) { b_bit |= 1u << i; }
    }

    std::array<float, kE8Sub> v1{};
    if (type_a_score >= type_b_score) {
        result.root = type_a_code;
        v1[static_cast<std::size_t>(best_i)] = s_i_bit ? 1.0f : -1.0f;
        v1[static_cast<std::size_t>(best_j)] = s_j_bit ? 1.0f : -1.0f;
    } else {
        result.root = static_cast<std::uint8_t>(112 + b_bit);
        for (int i = 0; i < kE8Sub; ++i) {
            v1[static_cast<std::size_t>(i)] =
                b_sign[static_cast<std::size_t>(i)] > 0 ? 0.5f : -0.5f;
        }
    }
    {
        const float span = std::max(std::abs(type_a_score), std::abs(type_b_score));
        if (std::abs(type_a_score - type_b_score) <= 1e-6f * std::max(span, 1.0f)) {
            result.ambiguous = true;
        }
    }

    // Residual hyperoctahedral axis: the largest coordinate of u after the root is removed.
    constexpr float kInvSqrt2 = 0.7071067811865475f;
    std::array<float, kE8Sub> scaled_v1{};
    std::array<float, kE8Sub> dot{};
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index = static_cast<std::size_t>(i);
        scaled_v1[index] = v1[index] * kInvSqrt2;
        dot[index]       = u[index] * scaled_v1[index];
    }
    butterfly_add(dot);
    std::array<float, kE8Sub> res{};
    std::array<float, kE8Sub> abs_res{};
    for (int i = 0; i < kE8Sub; ++i) {
        const auto index      = static_cast<std::size_t>(i);
        const float projection = dot[0] * scaled_v1[index];
        res[index]             = u[index] - projection;
        abs_res[index]         = std::abs(res[index]);
    }
    std::array<float, kE8Sub> max_res = abs_res;
    std::array<int, kE8Sub> best_dim{};
    for (int i = 0; i < kE8Sub; ++i) { best_dim[static_cast<std::size_t>(i)] = i; }
    for (int mask = 4; mask > 0; mask >>= 1) {
        auto nv = max_res;
        auto ni = best_dim;
        for (int i = 0; i < kE8Sub; ++i) {
            const auto self    = static_cast<std::size_t>(i);
            const auto partner = static_cast<std::size_t>(i ^ mask);
            const float other  = max_res[partner];
            const int other_id = best_dim[partner];
            if (other > max_res[self] || (other == max_res[self] && other_id < best_dim[self])) {
                nv[self] = other;
                ni[self] = other_id;
            }
        }
        max_res  = nv;
        best_dim = ni;
    }
    {
        // A near-tie between the two largest residual coordinates can flip the stored axis.
        float first  = -1.0f;
        float second = -1.0f;
        for (const float a : abs_res) {
            if (a > first) {
                second = first;
                first  = a;
            } else if (a > second) {
                second = a;
            }
        }
        if (first - second <= 1e-6f * std::max(first, 1.0f)) { result.ambiguous = true; }
    }
    const int axis_dim   = best_dim[0];
    const unsigned sign_bit = res[static_cast<std::size_t>(axis_dim)] >= 0.0f ? 0u : 1u;
    const unsigned axis_idx = (static_cast<unsigned>(axis_dim) << 1) | sign_bit;

    if (rad_idx == 0) {
        result.root     = 0;
        result.rad_axis = 0;
    } else {
        result.rad_axis = static_cast<std::uint8_t>((rad_idx << 4) | (axis_idx & 0x0fu));
    }
    return result;
}

} // namespace ninfer::test::rkv
