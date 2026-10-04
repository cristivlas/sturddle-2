#pragma once
/*
 * Extend vectorclass with int8 dot products, which VCL2 does not provide.
 */
#include "vectorclass.h"

/*
 * acc + sums of 4 adjacent u8 x s8 products per int32 lane.
 * Without VNNI, maddubs saturates adjacent pairs to int16: keep a <= 127.
 */
#if INSTRSET >= 10
static inline Vec16i dot_add(Vec16i const acc, Vec64uc const a, Vec64c const b)
{
  #if __AVX512VNNI__
    return _mm512_dpbusd_epi32(acc, a, b);
  #else
    return acc + Vec16i(_mm512_madd_epi16(_mm512_maddubs_epi16(a, b), _mm512_set1_epi16(1)));
  #endif /* __AVX512VNNI__ */
}
#endif /* INSTRSET >= 10 */

#if INSTRSET >= 8
static inline Vec8i dot_add(Vec8i const acc, Vec32uc const a, Vec32c const b)
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

static inline Vec4i dot_add(Vec4i const acc, Vec16uc const a, Vec16c const b)
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
