#pragma once
/*
 * Replicate vectorclass parts to get nnue.h to compile on ARM.
 */
#define SIMDE_ENABLE_NATIVE_ALIASES

#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"

#if (__arm64__) || (__aarch64__)
    #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC /* clang -march=armv8.2-a+fp16 */
        #define INSTRSET 8 /* use Vec8f half-precision implementation */
        #define ARCH "ARM64+FP16"
    #else
        #define ARCH "ARM64"
    #endif
#else
    #define ARCH "ARM"
#endif


#if 1 || SIMD_EMULATION /* emulate with SIMDE */

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


class Vec8s /* Emulate with SIMDE */
{
    __m128i xmm;

public:
    static constexpr size_t size() { return 8; }

    Vec8s() = default;

    Vec8s(__m128i x) : xmm(x) {}

    Vec8s(int i) { xmm = _mm_set1_epi16((int16_t)i); }

    INLINE operator __m128i() const { return xmm; }

    INLINE Vec8s & load_a(void const * p)
    {
        xmm = _mm_load_si128((__m128i const*)p);
        return *this;
    }
};


INLINE Vec8s max(Vec8s a, Vec8s b)
{
    return _mm_max_epi16(a, b);
}


INLINE __m256i extend(Vec8s a)
{
    return _mm256_cvtepi16_epi32(a);
}


// Extend the low 4 elements to 32 bits with sign extension
INLINE __m128i extend_low(Vec8s a)
{
    __m128i sign = _mm_srai_epi16(a, 15);   // sign bit
    return _mm_unpacklo_epi16(a, sign);     // interleave with sign extensions
}


// Extend the high 4 elements to 32 bits with sign extension
INLINE __m128i extend_high(Vec8s a)
{
    __m128i sign = _mm_srai_epi16(a, 15);   // sign bit
    return _mm_unpackhi_epi16(a, sign);     // interleave with sign extensions
}


INLINE Vec4f to_float(__m128i a)
{
    return _mm_cvtepi32_ps(a);
}


INLINE int32_t horizontal_add (__m256i a)
{
    // Add upper and lower 128-bit lanes: [a7+a3, a6+a2, a5+a1, a4+a0]
    __m128i sum1  = _mm_add_epi32(_mm256_extracti128_si256(a, 1), _mm256_castsi256_si128(a));

    // Add with high 64-bit part: [(a7+a3)+(a5+a1), (a6+a2)+(a4+a0), (a5+a1), (a4+a0)]
    __m128i sum2  = _mm_add_epi32(sum1,_mm_unpackhi_epi64(sum1, sum1));

    // Add with shuffled version to get final sum in lowest 32 bits
    __m128i sum3  = _mm_add_epi32(sum2,_mm_shuffle_epi32(sum2, 1));
    return (int32_t)_mm_cvtsi128_si32(sum3);
}


#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/* Vec8f is currently supported on FP16 Neon only */
INLINE Vec8f to_float(__m256i a)
{
    const auto m = _mm256_cvtepi32_ps(a);
    return Vec8f(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */


#if 0 && !__APPLE__ /* Emulate with SIMDE */
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
    __m128i sum1 = _mm_add_epi16(_mm256_extracti128_si256(a,1),_mm256_castsi256_si128(a));
    __m128i sum2 = _mm_add_epi16(sum1,_mm_unpackhi_epi64(sum1,sum1));
    __m128i sum3 = _mm_add_epi16(sum2,_mm_shuffle_epi32(sum2,1));
    __m128i sum4 = _mm_add_epi16(sum3,_mm_shufflelo_epi16(sum3,1));
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

INLINE Vec16s Vec16s max(Vec16s a, Vec16s b)
{
    return _mm256_max_epi16(a,b);
}

#else /* NEON */

class Vec16s
{
    int16x8x2_t _data;

public:
    static constexpr size_t size() { return 16; }

    Vec16s() = default;

    Vec16s(int16_t i) : _data{vdupq_n_s16(i), vdupq_n_s16(i)} {}

    Vec16s(int16x8x2_t data) : _data(data) {}

    Vec16s(int16_t i0, int16_t i1, int16_t i2,  int16_t i3,  int16_t i4,  int16_t i5,  int16_t i6,  int16_t i7,
           int16_t i8, int16_t i9, int16_t i10, int16_t i11, int16_t i12, int16_t i13, int16_t i14, int16_t i15)
    {
        int16_t values[16] = {i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15};
        _data = {vld1q_s16(&values[0]), vld1q_s16(&values[8])};
    }

    INLINE int16x8_t get_low() const { return _data.val[0]; }
    INLINE int16x8_t get_high() const { return _data.val[1]; }

    INLINE void load(const int16_t* p) { _data = vld1q_s16_x2(p); }
    INLINE void load_a(const int16_t* p) { _data = vld1q_s16_x2(p); }
    INLINE void store_a(int16_t* p) const { vst1q_s16_x2(p, _data); }
};

INLINE int16_t horizontal_add(const Vec16s& a)
{
    int16x8_t sum1 = vpaddq_s16(a.get_low(), a.get_high());
    int16x4_t sum2 = vadd_s16(vget_low_s16(sum1), vget_high_s16(sum1));
    int16x4_t sum3 = vpadd_s16(sum2, sum2);
    return vget_lane_s16(sum3, 0) + vget_lane_s16(sum3, 1);
}

INLINE bool horizontal_or(const Vec16s& a)
{
    int16x8_t or1 = vorrq_s16(a.get_low(), a.get_high());
    int16x4_t or2 = vorr_s16(vget_low_s16(or1), vget_high_s16(or1));
    int16x4_t or3 = vpmax_s16(or2, or2);
    return vget_lane_s16(or3, 0) | vget_lane_s16(or3, 1);
}

INLINE Vec16s operator + (const Vec16s& a, const Vec16s& b)
{
    int16x8x2_t result;
    result.val[0] = vaddq_s16(a.get_low(), b.get_low());
    result.val[1] = vaddq_s16(a.get_high(), b.get_high());
    return result;
}

INLINE Vec16s operator - (const Vec16s& a, const Vec16s& b)
{
    int16x8x2_t result;
    result.val[0] = vsubq_s16(a.get_low(), b.get_low());
    result.val[1] = vsubq_s16(a.get_high(), b.get_high());
    return result;
}

INLINE Vec16s operator * (const Vec16s& a, const Vec16s& b)
{
    int16x8x2_t result;
    result.val[0] = vmulq_s16(a.get_low(), b.get_low());
    result.val[1] = vmulq_s16(a.get_high(), b.get_high());
    return result;
}


INLINE Vec16s max(const Vec16s& a, const Vec16s& b)
{
    int16x8x2_t result;
    result.val[0] = vmaxq_s16(a.get_low(), b.get_low());
    result.val[1] = vmaxq_s16(a.get_high(),b.get_high());
    return result;
}

template <size_t N>
INLINE Vec16s shift_right(const Vec16s& a)
{
    int16x8x2_t result;
    result.val[0] = vshrq_n_s16(a.get_low(), N);
    result.val[1] = vshrq_n_s16(a.get_high(), N);
    return result;
}

#endif /* !__APPLE__ */


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
