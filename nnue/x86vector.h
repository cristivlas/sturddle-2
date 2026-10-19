#pragma once
/*
 * Extend vectorclass with int8 dot products, int16 mul_hi and lane compression, which VCL2 does not provide.
 */
#include <bit>
#include <cstddef>
#include <cstdint>
#include "vectorclass.h"

#if defined(__AVX512BF16__) && defined(__AVX512VL__)
    #define USE_BF16 true
#else
    #define USE_BF16 false
#endif

/* Positions of the set bits of each W-bit mask, lowest first */
template <int W> struct SetBitsTable
{
    int32_t index[1 << W][W] = { };

    constexpr SetBitsTable()
    {
        for (int m = 0; m != (1 << W); ++m)
            for (int k = 0, n = 0; k != W; ++k)
                if ((m >> k) & 1)
                    index[m][n++] = k;
    }
};

/* Append base + k for each non-zero lane k of v at out[count]; count <= base keeps the full-vector store in bounds */
#if INSTRSET >= 10
template <size_t N> INLINE int append_nonzero_lanes(int32_t (&out)[N], int count, int base, Vec16i const v)
{
    const auto mask = to_bits(v != 0);
    const Vec16i lanes = Vec16i(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15) + base;
    _mm512_storeu_si512(&out[count], _mm512_maskz_compress_epi32(mask, lanes));
    return count + std::popcount(unsigned(mask));
}
#endif /* INSTRSET >= 10 */

#if INSTRSET >= 8
inline constexpr SetBitsTable<8> SET_BITS_8;

template <size_t N> INLINE int append_nonzero_lanes(int32_t (&out)[N], int count, int base, Vec8i const v)
{
    const auto mask = to_bits(v != 0);
    (Vec8i().load(SET_BITS_8.index[mask]) + base).store(&out[count]);
    return count + std::popcount(unsigned(mask));
}
#endif /* INSTRSET >= 8 */

inline constexpr SetBitsTable<4> SET_BITS_4;

template <size_t N> INLINE int append_nonzero_lanes(int32_t (&out)[N], int count, int base, Vec4i const v)
{
    const auto mask = to_bits(v != 0);
    (Vec4i().load(SET_BITS_4.index[mask]) + base).store(&out[count]);
    return count + std::popcount(unsigned(mask));
}

/* Saturate signed 16-bit to unsigned 8-bit, low then high, in order: negatives become 0 */
#if INSTRSET >= 10
INLINE Vec64uc compress_saturated_s2u(Vec32s const low, Vec32s const high)
{
    return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), _mm512_packus_epi16(low, high));
}
#endif /* INSTRSET >= 10 */

#if INSTRSET >= 8
INLINE Vec32uc compress_saturated_s2u(Vec16s const low, Vec16s const high)
{
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(low, high), 0xD8);
}
#endif /* INSTRSET >= 8 */

INLINE Vec16uc compress_saturated_s2u(Vec8s const low, Vec8s const high)
{
    return _mm_packus_epi16(low, high);
}

/** High 16 bits of the signed 16 x 16 products */
#if INSTRSET >= 10
INLINE Vec32s mul_hi(Vec32s const a, Vec32s const b)
{
    return _mm512_mulhi_epi16(a, b);
}
#endif /* INSTRSET >= 10 */

#if INSTRSET >= 8
INLINE Vec16s mul_hi(Vec16s const a, Vec16s const b)
{
    return _mm256_mulhi_epi16(a, b);
}
#endif /* INSTRSET >= 8 */

INLINE Vec8s mul_hi(Vec8s const a, Vec8s const b)
{
    return _mm_mulhi_epi16(a, b);
}

/**
 * Multiply bytes (u8 x s8), adding each group of 4 products into one int32 lane of acc.
 * Without VNNI, pairs are summed in int16 first, which saturates unless a <= 127.
 */
#if INSTRSET >= 10
INLINE Vec16i dot_add(Vec16i const acc, Vec64uc const a, Vec64c const b)
{
  #if __AVX512VNNI__
    return _mm512_dpbusd_epi32(acc, a, b);
  #else
    return acc + Vec16i(_mm512_madd_epi16(_mm512_maddubs_epi16(a, b), _mm512_set1_epi16(1)));
  #endif /* __AVX512VNNI__ */
}
#endif /* INSTRSET >= 10 */

#if INSTRSET >= 8
INLINE Vec8i dot_add(Vec8i const acc, Vec32uc const a, Vec32c const b)
{
  #if __AVXVNNI__
    return _mm256_dpbusd_avx_epi32(acc, a, b);
  #elif __AVX512VNNI__ && __AVX512VL__
    return _mm256_dpbusd_epi32(acc, a, b);
  #else
    return acc + Vec8i(_mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1)));
  #endif /* __AVXVNNI__ */
}
#endif /* INSTRSET >= 8 */

INLINE Vec4i dot_add(Vec4i const acc, Vec16uc const a, Vec16c const b)
{
#if __AVXVNNI__
    return _mm_dpbusd_avx_epi32(acc, a, b);
#elif __AVX512VNNI__ && __AVX512VL__
    return _mm_dpbusd_epi32(acc, a, b);
#elif INSTRSET >= 4 /* SSSE3 */
    return acc + Vec4i(_mm_madd_epi16(_mm_maddubs_epi16(a, b), _mm_set1_epi16(1)));
#else
    const __m128i zero = _mm_setzero_si128();
    const __m128i a_lo = _mm_unpacklo_epi8(a, zero), a_hi = _mm_unpackhi_epi8(a, zero);
    const __m128i b_lo = _mm_srai_epi16(_mm_unpacklo_epi8(b, b), 8), b_hi = _mm_srai_epi16(_mm_unpackhi_epi8(b, b), 8);
    /* pair sums of bytes 0..7 and 8..15; adding even and odd pairs gives 4 adjacent bytes per lane */
    const __m128 lo = _mm_castsi128_ps(_mm_madd_epi16(a_lo, b_lo)), hi = _mm_castsi128_ps(_mm_madd_epi16(a_hi, b_hi));
    const __m128i even = _mm_castps_si128(_mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i odd = _mm_castps_si128(_mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1)));
    return acc + Vec4i(_mm_add_epi32(even, odd));
#endif /* __AVXVNNI__ */
}


#if USE_BF16
    class Vec32_bf16
    {
        __m512bh v;

    public:
        static constexpr size_t size() { return 32; }

        Vec32_bf16() = default;
        Vec32_bf16(__m512bh x) : v(x) {}

        operator __m512bh() const { return v; }

        INLINE void load_a(const float *p)
        {
            Vec16f low, high;
            low.load_a(p);
            high.load_a(p + 16);
            // float32 to bf16 using dedicated conversion instruction (with rounding)
            v = _mm512_cvtne2ps_pbh(high, low);
        }

        INLINE void load_a(const __bf16 *p)
        {
            v = (__m512bh)_mm512_load_si512((const __m512i*)p);
        }
    };

    INLINE Vec16f mul_add(const Vec32_bf16& a, const Vec32_bf16& b, Vec16f acc)
    {
        return Vec16f(_mm512_dpbf16_ps(acc, a, b));
    }

    template <int N> INLINE void load_partial(Vec16f& v, const __bf16* p)
    {
        if constexpr (N == Vec16f::size())
        {
            __m256bh vh = (__m256bh)_mm256_load_si256((const __m256i*)p);
            // bf16 to float32: zero-extend to 32 bits, shift left by 16
            v = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(vh), 16));
        }
        else
        {
            static_assert(false);
        }
    }
#endif /* USE_BF16 */

#if 0 /* unused since the sparse int8 dot */
/**
 * Sum up each of the 8 vectors: adding them pairwise is cheaper than 8 separate reductions.
 */
INLINE void horizontal_add(Vec4i const (&v)[8], int32_t (&out)[8])
{
    const auto pair32 = [](__m128i a, __m128i b) { return _mm_add_epi32(_mm_unpacklo_epi32(a, b), _mm_unpackhi_epi32(a, b)); };
    const auto pair64 = [](__m128i a, __m128i b) { return _mm_add_epi32(_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b)); };
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), pair64(pair32(v[0], v[1]), pair32(v[2], v[3])));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[4]), pair64(pair32(v[4], v[5]), pair32(v[6], v[7])));
}

#if INSTRSET >= 8
INLINE void horizontal_add(Vec8i const (&v)[8], int32_t (&out)[8])
{
    const auto pair32 = [](__m256i a, __m256i b) { return _mm256_add_epi32(_mm256_unpacklo_epi32(a, b), _mm256_unpackhi_epi32(a, b)); };
    const auto pair64 = [](__m256i a, __m256i b) { return _mm256_add_epi32(_mm256_unpacklo_epi64(a, b), _mm256_unpackhi_epi64(a, b)); };
    /* each 128-bit half holds partial sums; adding the halves finishes the job */
    const __m256i lo = pair64(pair32(v[0], v[1]), pair32(v[2], v[3]));
    const __m256i hi = pair64(pair32(v[4], v[5]), pair32(v[6], v[7]));
    const __m256i r = _mm256_add_epi32(_mm256_permute2x128_si256(lo, hi, 0x20), _mm256_permute2x128_si256(lo, hi, 0x31));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), r);
}
#endif /* INSTRSET >= 8 */

#if INSTRSET >= 10
INLINE void horizontal_add(Vec16i const (&v)[8], int32_t (&out)[8])
{
    Vec8i h[8];
    for (int k = 0; k != 8; ++k)
        h[k] = v[k].get_low() + v[k].get_high();
    horizontal_add(h, out);
}
#endif /* INSTRSET >= 10 */
#endif /* unused */
