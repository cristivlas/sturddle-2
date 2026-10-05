#pragma once
/*
 * Extend vectorclass with int8 dot products, which VCL2 does not provide.
 */
#include "vectorclass.h"

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
    return acc + Vec4i(_mm_madd_epi16(a_lo, b_lo)) + Vec4i(_mm_madd_epi16(a_hi, b_hi));
#endif /* __AVXVNNI__ */
}

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
