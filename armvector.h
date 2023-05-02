#pragma once
/*
 * Replicate parts of vectorclass to get nnue.h to compile on arm.
 */
#define SIMDE_ENABLE_NATIVE_ALIASES

#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"

class Vec4f
{
    __m128 xmm;

public:
    static constexpr size_t size() { return 4; }

    Vec4f() = default;
    Vec4f(float f) : xmm(_mm_set1_ps(f)) {}
    Vec4f(__m128 x) : xmm(x) {}

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

INLINE Vec4f& operator += (Vec4f & a, Vec4f b)
{
    a = a + b;
    return a;
}

INLINE Vec4f operator - (Vec4f a, Vec4f b)
{
    return _mm_sub_ps(a, b);
}

INLINE Vec4f& operator -= (Vec4f & a, Vec4f b)
{
    a = a - b;
    return a;
}

INLINE Vec4f operator * (Vec4f a, Vec4f b)
{
    return _mm_mul_ps(a, b);
}

INLINE Vec4f mul_add(Vec4f a, Vec4f b, Vec4f c)
{
    return _mm_fmadd_ps(a, b, c);
}

#if __aarch64__
class Vec16s
{
    __m256i ymm;

public:
    static constexpr size_t size() { return 16; }

    Vec16s() = default;
    Vec16s(int16_t i) : ymm(_mm256_set1_epi16(i)) {}
    Vec16s(__m256i y) : ymm(y) {}

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

#else
class Vec16s
{
    __m128i y0; // low half
    __m128i y1; // high half

public:
    static constexpr size_t size() { return 16; }

    Vec16s() = default;
    Vec16s(int16_t i) : y0(_mm_set1_epi16(i)), y1(_mm_set1_epi16(i)) {}
    Vec16s(__m128i a0, __m128i a1) : y0(a0), y1(a1) {}

    void load(const int16_t* p)
    {
        y0 = _mm_loadu_si128((__m128i const*)p);
        y1 = _mm_loadu_si128((__m128i const*)p + 1);
    }

    void load_a(const int16_t* p)
    {
        y0 = _mm_load_si128((__m128i const*)p);
        y1 = _mm_load_si128((__m128i const*)p + 1);
    }

    void store_a(int16_t* p) const
    {
        _mm_store_si128((__m128i*)p,     y0);
        _mm_store_si128((__m128i*)p + 1, y1);
    }

    __m128i get_low() const { return y0; }
    __m128i get_high() const { return y1; }
};

INLINE bool horizontal_or(const Vec16s a)
{
    auto b = _mm_or_si128(a.get_low(), a.get_high());
    return !_mm_testz_si128(b, b);
}

INLINE Vec16s operator + (Vec16s a, Vec16s b)
{
    return Vec16s(
        _mm_add_epi16(a.get_low(), b.get_low()),
        _mm_add_epi16(a.get_high(), b.get_high())
    );
}

INLINE Vec16s operator - (Vec16s a, Vec16s b)
{
    return Vec16s(
        _mm_sub_epi16(a.get_low(), b.get_low()),
        _mm_sub_epi16(a.get_high(), b.get_high())
    );
}

INLINE Vec16s operator * (Vec16s a, Vec16s b)
{
    return Vec16s(
        _mm_mullo_epi16(a.get_low(), b.get_low()),
        _mm_mullo_epi16(a.get_high(), b.get_high())
    );
}

#endif /* !__aarch64__ */

INLINE Vec16s& operator += (Vec16s& a, Vec16s b)
{
    a = a + b;
    return a;
}

INLINE Vec16s& operator -= (Vec16s& a, Vec16s b)
{
    a = a - b;
    return a;
}

