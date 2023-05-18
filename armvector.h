#pragma once
/*
 * Replicate vectorclass parts to get nnue.h to compile on ARM.
 */
#define SIMDE_ENABLE_NATIVE_ALIASES
// #define INSTRSET 8 /* emulate AVX2 */

#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"

// Join two 128-bit vectors.
#define set_m128r(lo,hi) _mm256_insertf128_ps(_mm256_castps128_ps256(lo),(hi),1)


class Vec4f
{
    __m128 xmm;

public:
    static constexpr size_t size() { return 4; }

    Vec4f() = default;
    Vec4f(float f) : xmm(_mm_set1_ps(f)) {}
    Vec4f(__m128 x) : xmm(x) {}
    Vec4f(float f0, float f1, float f2, float f3) {
        xmm = _mm_setr_ps(f0, f1, f2, f3);
    }
    void load_a(const float* p) { xmm = _mm_load_ps(p); }
    void store_a(float* p) const { _mm_store_ps(p, xmm); }

    operator __m128() const { return xmm; }
};

INLINE float horizontal_add(Vec4f a)
{
    __m128 t1 = _mm_movehl_ps(a, a);
    __m128 t2 = _mm_add_ps(a, t1);
    __m128 t3 = _mm_shuffle_ps(t2, t2, 1);
    __m128 t4 = _mm_add_ss(t2, t3);
    return _mm_cvtss_f32(t4);
}

INLINE Vec4f operator + (Vec4f a, Vec4f b)
{
    return _mm_add_ps(a, b);
}

INLINE Vec4f operator - (Vec4f a, Vec4f b)
{
    return _mm_sub_ps(a, b);
}

INLINE Vec4f operator * (Vec4f a, Vec4f b)
{
    return _mm_mul_ps(a, b);
}

INLINE Vec4f mul_add(Vec4f a, Vec4f b, Vec4f c)
{
    return _mm_fmadd_ps(a, b, c);
}

INLINE Vec4f max(Vec4f a, Vec4f b)
{
    return _mm_max_ps(a, b);
}


class Vec8f
{
    __m256 xmm;

public:
    static constexpr size_t size() { return 8; }

    Vec8f() = default;
    Vec8f(float f) : xmm(_mm256_set1_ps(f)) {}
    Vec8f(__m256 x) : xmm(x) {}
    Vec8f(Vec4f const a0, Vec4f const a1) {
        xmm = set_m128r(a0, a1);
    }
    Vec8f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
        xmm = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
    }

    void load_a(const float* p) { xmm = _mm256_load_ps(p); }
    void store_a(float* p) const { _mm256_store_ps(p, xmm); }

    operator __m256() const { return xmm; }
};

INLINE float horizontal_add(Vec8f a)
{
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(a, 1), _mm256_castps256_ps128(a));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

INLINE Vec8f operator + (Vec8f a, Vec8f b)
{
    return _mm256_add_ps(a, b);
}

INLINE Vec8f operator - (Vec8f a, Vec8f b)
{
    return _mm256_sub_ps(a, b);
}

INLINE Vec8f operator * (Vec8f a, Vec8f b)
{
    return _mm256_mul_ps(a, b);
}

INLINE Vec8f mul_add(Vec8f a, Vec8f b, Vec8f c)
{
    return _mm256_fmadd_ps(a, b, c);
}


class Vec16s
{
    __m256i ymm;

public:
    static constexpr size_t size() { return 16; }

    Vec16s() = default;
    Vec16s(int16_t i) : ymm(_mm256_set1_epi16(i)) {}
    Vec16s(__m256i y) : ymm(y) {}

    Vec16s(int16_t i0, int16_t i1, int16_t i2,  int16_t i3,  int16_t i4,  int16_t i5,  int16_t i6,  int16_t i7,
           int16_t i8, int16_t i9, int16_t i10, int16_t i11, int16_t i12, int16_t i13, int16_t i14, int16_t i15) {
        ymm = _mm256_setr_epi16(i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15 );
    }

    void load(const int16_t* p)
    {
        ymm = _mm256_loadu_si256((__m256i const*)p);
    }

    void load_a(const int16_t* p)
    {
        ymm = _mm256_load_si256((__m256i const*)p);
    }

    void store_a(int16_t* p) const
    {
        _mm256_store_si256((__m256i*)p, ymm);
    }

    operator __m256i() const { return ymm; }
};

INLINE int16_t horizontal_add(Vec16s a)
{
    __m128i sum1  = _mm_add_epi16(_mm256_extracti128_si256(a,1),_mm256_castsi256_si128(a));
    __m128i sum2  = _mm_add_epi16(sum1,_mm_unpackhi_epi64(sum1,sum1));
    __m128i sum3  = _mm_add_epi16(sum2,_mm_shuffle_epi32(sum2,1));
    __m128i sum4  = _mm_add_epi16(sum3,_mm_shufflelo_epi16(sum3,1));
    return (int16_t)_mm_cvtsi128_si32(sum4);
}

INLINE bool horizontal_or(Vec16s a)
{
    return !_mm256_testz_si256(a, a);
}

INLINE Vec16s operator + (Vec16s a, Vec16s b)
{
    return _mm256_add_epi16(a, b);
}

INLINE Vec16s operator - (Vec16s a, Vec16s b)
{
    return _mm256_sub_epi16(a, b);
}

INLINE Vec16s operator * (Vec16s a, Vec16s b)
{
    return _mm256_mullo_epi16(a, b);
}

INLINE Vec8f max(Vec8f a, Vec8f b)
{
    return _mm256_max_ps(a, b);
}

template <typename V> INLINE V& operator += (V& a, V b)
{
    a = a + b;
    return a;
}

template <typename V> INLINE V& operator -= (V& a, V b)
{
    a = a - b;
    return a;
}

