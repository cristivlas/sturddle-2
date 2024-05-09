#pragma once
/*
 * Replicate vectorclass parts to get nnue.h to compile on ARM.
 */
#define SIMDE_ENABLE_NATIVE_ALIASES

#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"

#if (__arm64__) || (__aarch64__)
    #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        #define INSTRSET 8 /* use Vec8f */
        #define ARCH "ARM64+FP16"
    #else
        #define ARCH "ARM64"
    #endif
#else
    #define ARCH "ARM"
#endif

#if SIMD_EMULATION /* emulate with SIMDE */

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

INLINE float horizontal_add(const Vec4f& a)
{
    __m128 t1 = _mm_movehl_ps(a, a);
    __m128 t2 = _mm_add_ps(a, t1);
    __m128 t3 = _mm_shuffle_ps(t2, t2, 1);
    __m128 t4 = _mm_add_ss(t2, t3);
    return _mm_cvtss_f32(t4);
}

INLINE Vec4f operator + (const Vec4f& a, const Vec4f& b)
{
    return _mm_add_ps(a, b);
}

INLINE Vec4f operator - (const Vec4f& a, const Vec4f& b)
{
    return _mm_sub_ps(a, b);
}

INLINE Vec4f operator * (const Vec4f& a, const Vec4f& b)
{
    return _mm_mul_ps(a, b);
}

INLINE Vec4f mul_add(const Vec4f& a, const Vec4f& b, const Vec4f& c)
{
    return _mm_fmadd_ps(a, b, c);
}

INLINE Vec4f max(const Vec4f& a, const Vec4f& b)
{
    return _mm_max_ps(a, b);
}

#else /* NEON */

class Vec4f
{
    float32x4_t v;

public:
    static constexpr size_t size() { return 4; }

    Vec4f() = default;
    Vec4f(float f) : v(vdupq_n_f32(f)) {}
    Vec4f(float32x4_t x) : v(x) {}
    Vec4f(float f0, float f1, float f2, float f3) {
        float f[4] = { f0, f1, f2, f3 };
        load_a(f);
    }
    void load_a(const float* p) { v = vld1q_f32(p); }
    void store_a(float* p) const { vst1q_f32(p, v); }

    operator float32x4_t() const { return v; }
};

INLINE float horizontal_add(const Vec4f& a)
{
#if (__arm64__) || (__aarch64__)
    return vaddvq_f32(a);
#else
    // Pairwise add the elements, reducing the vector to half its original size
    float32x2_t vsum = vadd_f32(vget_high_f32(a), vget_low_f32(a));

    // Use the vpadd instruction to add the remaining elements
    vsum = vpadd_f32(vsum, vsum);

    // Extract the sum from the vector
    return vget_lane_f32(vsum, 0);
#endif
}

INLINE Vec4f operator + (const Vec4f& a, const Vec4f& b)
{
    return vaddq_f32(a, b);
}

INLINE Vec4f operator - (const Vec4f& a, const Vec4f& b)
{
    return vsubq_f32(a, b);
}

INLINE Vec4f operator * (const Vec4f& a, const Vec4f& b)
{
    return vmulq_f32(a, b);
}

INLINE Vec4f mul_add(const Vec4f& a, const Vec4f& b, const Vec4f& c)
{
    return vmlaq_f32(c, a, b); // may map to vfmaq_f32 on some archs?
}

INLINE Vec4f max(const Vec4f& a, const Vec4f& b)
{
    return vmaxq_f32(a, b);
}
#endif /* NEON */


#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

/* Implement Vec8f using half precision */
class Vec8f
{
    float16x8_t v;

public:
    static constexpr size_t size() { return 8; }

    Vec8f() = default;
    Vec8f(float f) : v(vdupq_n_f16(f)) {}
    Vec8f(float16x8_t x) : v(x) {}
    Vec8f(float32x4_t v1, float32x4_t v2) : v(vcombine_f16(vcvt_f16_f32(v1), vcvt_f16_f32(v2))) {}
    Vec8f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
        const float f[8] = { f0, f1, f2, f3, f4, f5, f6, f7 };
        load_a(f);
    }

    INLINE void load_a(const float* p)
    {
        const float32x4_t v1(vld1q_f32(p));
        const float32x4_t v2(vld1q_f32(p + 4));
        v = vcombine_f16(vcvt_f16_f32(v1), vcvt_f16_f32(v2));
    }

    INLINE void store_a(float* p) const
    {
        float32x4_t v1(vcvt_f32_f16(vget_low_f16(v)));
        float32x4_t v2(vcvt_f32_f16(vget_high_f16(v)));
        vst1q_f32(p, v1);
        vst1q_f32(p + 4, v2);
    }

    INLINE float operator[](int n) const { return v[n]; }

    INLINE operator float16x8_t() const { return v; }
};

INLINE float horizontal_add(const Vec8f& x)
{
#if 0
    // Split and add
    float16x4_t a = vadd_f16(vget_high_f16(x), vget_low_f16(x));

    // Split and add again
    float16x4_t b = vdup_n_f16(vget_lane_f16(a, 0) + vget_lane_f16(a, 1) +
                               vget_lane_f16(a, 2) + vget_lane_f16(a, 3));

    // Return the result (all lanes are the same)
    return vget_lane_f16(b, 0);
#else 
    // Pairwise addition reduces the vector in steps
    float16x4_t sum = vpadd_f16(vget_low_f16(x), vget_high_f16(x));
    sum = vpadd_f16(sum, sum);
    sum = vpadd_f16(sum, sum);

    // Return the result (first element in the vector)
    return vget_lane_f16(sum, 0);
#endif
}

INLINE Vec8f operator + (const Vec8f& a, const Vec8f& b)
{
    return vaddq_f16(a, b);
}

INLINE Vec8f operator - (const Vec8f& a, const Vec8f& b)
{
    return vsubq_f16(a, b);
}

INLINE Vec8f operator * (const Vec8f& a, const Vec8f& b)
{
    return vmulq_f16(a, b);
}

INLINE Vec8f max(const Vec8f& a, const Vec8f& b)
{
    return vmaxq_f16(a, b);
}

INLINE Vec8f mul_add(const Vec8f& a, const Vec8f& b, const Vec8f& c)
{
    return vfmaq_f16(c, a, b);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */


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

INLINE int16_t horizontal_add(const Vec16s& a)
{
    __m128i sum1  = _mm_add_epi16(_mm256_extracti128_si256(a,1),_mm256_castsi256_si128(a));
    __m128i sum2  = _mm_add_epi16(sum1,_mm_unpackhi_epi64(sum1,sum1));
    __m128i sum3  = _mm_add_epi16(sum2,_mm_shuffle_epi32(sum2,1));
    __m128i sum4  = _mm_add_epi16(sum3,_mm_shufflelo_epi16(sum3,1));
    return (int16_t)_mm_cvtsi128_si32(sum4);
}

INLINE bool horizontal_or(const Vec16s& a)
{
    return !_mm256_testz_si256(a, a);
}

INLINE Vec16s operator + (const Vec16s& a, const Vec16s& b)
{
    return _mm256_add_epi16(a, b);
}

INLINE Vec16s operator - (const Vec16s& a, const Vec16s& b)
{
    return _mm256_sub_epi16(a, b);
}

INLINE Vec16s operator * (const Vec16s& a, const Vec16s& b)
{
    return _mm256_mullo_epi16(a, b);
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
