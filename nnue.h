#pragma once
/*
 * Sturddle Chess Engine (C) 2023 - 2026 Cristian Vlasceanu
 * --------------------------------------------------------------------------
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * --------------------------------------------------------------------------
 * Third-party files included in this project are subject to copyright
 * and licensed as stated in their respective header notes.
 * --------------------------------------------------------------------------
 */
#include "common.h"
#include "chess.h"
#include <fstream>
#include <tuple>

#if (__amd64__) || (__x86_64__) || (__i386__) || (_M_AMD64) || (_M_X64) || (_M_IX86)
    #include "vectorclass.h"
#elif (__arm__) || (__arm64__) || (__aarch64__)
    #define __ARM__ true
    #include "armvector.h"
#endif

#if __AVXVNNI__
    #define ARCH_VNNI "/VNNI"
#else
    #define ARCH_VNNI
#endif /* __AVXVNNI__ */

#ifndef ARCH
    #if INSTRSET >= 9 /* AVX 512 */
        #define ARCH "AVX512"
    #elif INSTRSET >= 8
        #define ARCH "AVX2"
    #elif INSTRSET >= 7
        #define ARCH "AVX"
    #elif INSTRSET >= 4
        #define ARCH "SSSE3"
    #else
        #define ARCH "SSE2"
    #endif /* INSTRSET*/
#endif /* ARCH */

#define ALIGN alignas(64)

#ifndef DEBUG_INCREMENTAL
  #define DEBUG_INCREMENTAL false
#endif

/* CFLAGS=-march="armv8.2-a+fp16+dotprod" */
#if !__ARM__ || defined(__ARM_FEATURE_DOTPROD)
    #define USE_8_BIT_QUANTIZATION true
#endif


namespace nnue
{
    using namespace chess;
    using input_t = int16_t;

#if INSTRSET >= 9 /* AVX 512 */
    constexpr int INPUT_STRIDE = 32;
#else
    constexpr int INPUT_STRIDE = 16;
#endif

    constexpr int MAX_PIECES = 32;

    /** Feature encoding */
    constexpr int NUM_BUCKETS = 32;
    constexpr int PIECE_FEATURES = 768;
    constexpr int PERSPECTIVE_SIZE = NUM_BUCKETS * PIECE_FEATURES;

    constexpr int QSCALE_16 = 255;
    constexpr int QSCALE_8 = 64;
    constexpr int CLIP_RELU = 127;
    constexpr int EVAL_SCALE = 16;

    #if 0 && INSTRSET >= 9 /* AVX512 */
        using Vector = Vec16f;

        INLINE Vector horizontal_add(const Vector (&v)[16])
        {
            return Vector(
                horizontal_add(v[0]), horizontal_add(v[1]), horizontal_add(v[2]), horizontal_add(v[3]),
                horizontal_add(v[4]), horizontal_add(v[5]), horizontal_add(v[6]), horizontal_add(v[7]),
                horizontal_add(v[8]), horizontal_add(v[9]), horizontal_add(v[10]),horizontal_add(v[11]),
                horizontal_add(v[12]),horizontal_add(v[13]),horizontal_add(v[14]),horizontal_add(v[15]));
        }
    #elif INSTRSET >= 7 /* AVX, AVX2 */
        using Vector = Vec8f;

        INLINE Vector horizontal_add(const Vector (&v)[8])
        {
            return Vector(
                horizontal_add(v[0]), horizontal_add(v[1]), horizontal_add(v[2]), horizontal_add(v[3]),
                horizontal_add(v[4]), horizontal_add(v[5]), horizontal_add(v[6]), horizontal_add(v[7]));
        }
    #else
        using Vector = Vec4f;

        INLINE Vector horizontal_add(const Vector (&v)[4])
        {
            return Vector(horizontal_add(v[0]), horizontal_add(v[1]), horizontal_add(v[2]), horizontal_add(v[3]));
        }
    #endif /* INSTRSET */

#ifdef __FMA__  /* support fused multiply+add? */
    static const std::string instrset = ARCH "/FMA" ARCH_VNNI;
#else
    static const std::string instrset = ARCH ARCH_VNNI;
#endif /* __FMA__ */

    static const Vector v_zero(0.0);


    template <typename V>
    INLINE bool all_zero(V v)
    {
        return !horizontal_or(v);
    }

    template <typename V>
    INLINE Vec16s horizontal_add(const V (&v)[16])
    {
        return Vec16s(
            horizontal_add_x(v[0]), horizontal_add_x(v[1]), horizontal_add_x(v[2]), horizontal_add_x(v[3]),
            horizontal_add_x(v[4]), horizontal_add_x(v[5]), horizontal_add_x(v[6]), horizontal_add_x(v[7]),
            horizontal_add_x(v[8]), horizontal_add_x(v[9]), horizontal_add_x(v[10]),horizontal_add_x(v[11]),
            horizontal_add_x(v[12]),horizontal_add_x(v[13]),horizontal_add_x(v[14]),horizontal_add_x(v[15]));
    }

#if !__ARM__ /* Vec32s not supported on NEON */
    template <>
    INLINE bool all_zero<Vec32s>(Vec32s v)
    {
        return !horizontal_or(v.get_high() | v.get_low());
    }

    INLINE Vec32s horizontal_add(const Vec32s (&v)[32])
    {
        return Vec32s(
            horizontal_add_x(v[0]),  horizontal_add_x(v[1]),  horizontal_add_x(v[2]),  horizontal_add_x(v[3]),
            horizontal_add_x(v[4]),  horizontal_add_x(v[5]),  horizontal_add_x(v[6]),  horizontal_add_x(v[7]),
            horizontal_add_x(v[8]),  horizontal_add_x(v[9]),  horizontal_add_x(v[10]), horizontal_add_x(v[11]),
            horizontal_add_x(v[12]), horizontal_add_x(v[13]), horizontal_add_x(v[14]), horizontal_add_x(v[15]),
            horizontal_add_x(v[16]), horizontal_add_x(v[17]), horizontal_add_x(v[18]), horizontal_add_x(v[19]),
            horizontal_add_x(v[20]), horizontal_add_x(v[21]), horizontal_add_x(v[22]), horizontal_add_x(v[23]),
            horizontal_add_x(v[24]), horizontal_add_x(v[25]), horizontal_add_x(v[26]), horizontal_add_x(v[27]),
            horizontal_add_x(v[28]), horizontal_add_x(v[29]), horizontal_add_x(v[30]), horizontal_add_x(v[31]));
    }
#endif /* !__ARM__ */

    INLINE Vector horizontal_add(const Vector (&v)[1])
    {
        return horizontal_add(v[0]);
    }

    template <int N> INLINE void load_partial(Vector& v, const float* p)
    {
        if constexpr (N == 1)
            #if INSTRSET >= 8
                v.load_partial(1, p);
            #elif INSTRSET >= 7
                v = Vector(_mm_load_ss(p), _mm_setzero_ps());
            #else
                v = _mm_load_ss(p);
            #endif
        else if constexpr (N == Vector::size())
            v.load_a(p);
        else
            ASSERT(false);
    }

    template <int N> INLINE void store_partial(const Vector& v, float* p)
    {
        if constexpr (N == 1)
            #if INSTRSET >= 8
                v.store_partial(1, p);
            #elif INSTRSET >= 7
                #if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    *p = v[0];
                #else
                    _mm_store_ss(p, _mm256_castps256_ps128(v));
                #endif
            #else
                _mm_store_ss(p, v);
            #endif
        else if constexpr (N == Vector::size())
            v.store_a(p);
        else
            ASSERT(false);
    }

    template <unsigned int N>
    constexpr unsigned int round_down(unsigned int x)
    {
        return (x / N) * N;
    }

    template <unsigned int N>
    constexpr unsigned int round_up(unsigned int x)
    {
        return ((x + N - 1) / N) * N;
    }


    /** Rectified Linear Unit (reLU) activation */
    template <typename V> INLINE V relu(V v) { return max(0, v); }
    template <typename V> INLINE V clipped_relu(V v) { return min(CLIP_RELU, relu(v)); }

#if !USE_8_BIT_QUANTIZATION
    /** Clipped ReLU with scaling. Version w/o quantization, for reference/testing/debugging. */
    template <int N>
    INLINE void activate(const int16_t (&input)[N], float (&output)[N])
    {
        /* Scale accumulator outputs */
        constexpr float QSCALE_RECIP = 1.0f / QSCALE_16;

#if __ARM__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        /* Vec8f supported only on FP16 (half-precision) Neon */
        #pragma clang loop vectorize(enable)
        for (int i = 0; i != N; ++i)
        {
            output[i] = std::max<float>(0, float(std::min<int>(CLIP_RELU, input[i])) * QSCALE_RECIP);
        }
#else
    #if INSTRSET < 9
        using VF = Vec8f;
        using VS = Vec8s;
    #else
        using VF = Vec16f;
        using VS = Vec16s;
    #endif /* AVX512 */

        static_assert(N % VF::size() == 0);

        const VF v_scale(QSCALE_RECIP);

        for (size_t i = 0; i < N; i += VF::size())
        {
            VF v = to_float(extend(clipped_relu(VS().load_a(&input[i]))));
            (v * v_scale).store_a(&output[i]);
        }
#endif /* __ARM__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    }

#elif __ARM__
    template <int N>
    INLINE void activate(const int16_t (&input)[N], int8_t (&output)[N])
    {
        constexpr int VECSIZE = 16;  // Process 16x int16 at a time
        static_assert(N % VECSIZE == 0);

        for (size_t i = 0; i < N; i += VECSIZE)
        {
            // Load 2x 8 int16 values
            int16x8_t v_lo = vld1q_s16(&input[i]);
            int16x8_t v_hi = vld1q_s16(&input[i + 8]);

            // Clipped ReLU on both
            int16x8_t zero = vdupq_n_s16(0);
            int16x8_t max_val = vdupq_n_s16(127);

            v_lo = vmaxq_s16(v_lo, zero);
            v_lo = vminq_s16(v_lo, max_val);
            v_hi = vmaxq_s16(v_hi, zero);
            v_hi = vminq_s16(v_hi, max_val);

            // Narrow both halves
            int8x8_t v_lo_i8 = vqmovn_s16(v_lo);
            int8x8_t v_hi_i8 = vqmovn_s16(v_hi);

            // Combine and store 16x int8
            int8x16_t result = vcombine_s8(v_lo_i8, v_hi_i8);
            vst1q_s8(&output[i], result);
        }
    }
#else
    template <int N>
    INLINE void activate(const int16_t (&input)[N], int8_t (&output)[N])
    {
    #if INSTRSET >= 9 /* AVX 512 */
        using VecShort = Vec32s;
    #else
        using VecShort = Vec16s;
    #endif
        static_assert(N % VecShort::size() == 0);

        VecShort v;

        for (size_t i = 0; i < N; i += VecShort::size())
        {
            v.load_a(&input[i]);
            v = clipped_relu(v);

            // const auto in_range = (v >= 0) & (v <= 127);
            // ASSERT(horizontal_and(in_range));

            // Just store it. The 8-bit quantized multiplication handles scaling.
            compress_saturated(v).store_a(&output[i]);
        }
    }

#if INSTRSET < 8
    static const __m128i ONE_EPI16 = _mm_set1_epi16(1);
#endif
#if INSTRSET >= 8 && !defined(__AVXVNNI__)
    static const __m256i ONE_EPI16_AVX = _mm256_set1_epi16(1);
#endif


    INLINE Vec8i mul_add(Vec32uc a, Vec32c b, Vec8i acc)
    {
    #if __AVXVNNI__
        acc = _mm256_dpbusd_epi32(acc, a, b);
    #elif INSTRSET < 8
        /* SSSE3 */
        // Multiply a * b and accumulate neighbouring outputs into int16 values
        __m128i product_lo = _mm_maddubs_epi16(a.get_low(), b.get_low());
        __m128i product_hi = _mm_maddubs_epi16(a.get_high(), b.get_high());

        // Multiply product by 1 (idempotent) and accumulate neighbouring outputs into int32 values
        product_lo = _mm_madd_epi16(product_lo, ONE_EPI16);
        product_hi = _mm_madd_epi16(product_hi, ONE_EPI16);

        // Add to the main int32 accumulator
        acc = Vec8i(_mm_add_epi32(acc.get_low(), product_lo), _mm_add_epi32(acc.get_high(), product_hi));
    #else
        /* AVX2 */
        __m256i product = _mm256_maddubs_epi16(a, b);

        product = _mm256_madd_epi16(product, ONE_EPI16_AVX);
        acc = _mm256_add_epi32(acc, product);
    #endif
        return acc;
    }
#endif /* USE_8_BIT_QUANTIZATION */


    template <int I, int O, typename T, int Scale, bool Incremental>
    struct BaseLayer
    {
        static constexpr int ROWS = I;
        static constexpr int COLS = O;
        static constexpr int INPUTS = round_up<INPUT_STRIDE>(I);
        static constexpr int OUTPUTS = O;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */
        ALIGN T _w[INPUTS][OUTPUTS]; /* weights - only in accumulator (incremental) layers */
    };


    template <int I, int O, typename T, int Scale>
    struct BaseLayer<I, O, T, Scale, false>
    {
        static constexpr int ROWS = I;
        static constexpr int COLS = O;
        static constexpr int INPUTS = (Scale == 1) ? I : round_up<INPUT_STRIDE>(I);
        static constexpr int OUTPUTS = O;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */
    };


    template <int I, int O, typename T=float, int Scale=1, bool Incremental=false>
    struct Layer : BaseLayer<I, O, T, Scale, Incremental>
    {
        using Base = BaseLayer<I, O, T, Scale, Incremental>;
        using Base::INPUTS;
        using Base::OUTPUTS;
        using Base::_b;
        using Base::_wt;

        Layer() = default;

        Layer(const float(&w)[I][OUTPUTS], const float(&b)[OUTPUTS])
        {
            set_weights(w, b);
        }

        static constexpr size_t param_count()
        {
            return (I + 1) * O;
        }

        void set_weights(const float(&w)[I][OUTPUTS], const float(&b)[OUTPUTS])
        {
            for (int j = 0; j != OUTPUTS; ++j)
                if constexpr (Scale == 1)
                    _b[j] = b[j];
                else
                    _b[j] = T(std::round(b[j] * Scale));

            for (int i = 0; i != I; ++i)
            {
                for (int j = 0; j != OUTPUTS; ++j)
                {
                    if constexpr (Scale == 1)
                        _wt[j][i] = w[i][j];
                    else
                        _wt[j][i] = T(std::round(w[i][j] * Scale));

                    if constexpr (Incremental)
                        this->_w[i][j] = _wt[j][i];
                }
            }
            /* padding, if needed */
            for (int i = I; i != INPUTS; ++i)
            {
                for (int j = 0; j != OUTPUTS; ++j)
                {
                    if constexpr (Incremental)
                        this->_w[i][j] = 0;
                    _wt[j][i] = 0;
                }
            }
        }

        void load_weights(std::ifstream& file)
        {
            auto w = std::make_unique<float[]>(I * OUTPUTS);
            auto b = std::make_unique<float[]>(OUTPUTS);

            file.read(reinterpret_cast<char*>(w.get()), I * OUTPUTS * sizeof(float));
            file.read(reinterpret_cast<char*>(b.get()), OUTPUTS * sizeof(float));

            set_weights(reinterpret_cast<float(&)[I][OUTPUTS]>(*(w.get())), reinterpret_cast<float(&)[OUTPUTS]>(*(b.get())));
        }

        /* input (16-bit quantization) */
        template<typename V>
        INLINE void dot(int king_bucket, const input_t (&input)[INPUTS], V (&output)[OUTPUTS]) const
        {
            #define MUL_ADD(a, b, s) s += a * b
        #if INSTRSET >= 9 /* AVX 512 */
            using VecShort = Vec32s;
            using VSum = Vec32s;
        #else
            using VecShort = Vec16s;
            #if __AVXVNNI__
                #undef MUL_ADD
                #define MUL_ADD(a, b, s) s = _mm256_dpwssd_epi32(s, a, b)
                using VSum = Vec8i;
            #else
                using VSum = Vec16s;
            #endif /* __AVXVNNI__ */
        #endif /* INSTRSET */

            constexpr auto N = VecShort::size();
            static_assert(N == INPUT_STRIDE);
            static_assert(OUTPUTS % N == 0);

            constexpr auto R = round_down<N>(INPUTS);
            static_assert(R == INPUTS); /* expect padded inputs */

        #if 0
            const int start_i = 0;
            const int end_i = R;
        #else
            /* Calculate the active bucket range */
            const int start_i = king_bucket * PIECE_FEATURES;
            const int end_i = start_i + PIECE_FEATURES;
        #endif

            VecShort in, vw;
            VSum sum[N]; /* accumulate partial sums */

            for (int j = 0; j != OUTPUTS; j += N)
            {
                #pragma unroll N
                for (int k = 0; k != N; ++k)
                    sum[k] = VSum(0);

                for (int i = start_i; i < end_i; i += N)
                {
                    in.load_a(input + i);
                    if (all_zero(in))
                        continue;

                    for (int k = 0; k != N; ++k)
                    {
                        vw.load_a(&_wt[j + k][i]);
                        MUL_ADD(in, vw, sum[k]);
                    }
                }

                vw.load_a(&_b[j]);
                vw += horizontal_add(sum);
                vw.store_a(&output[j]);
            }
            #undef MUL_ADD
        }

    #if USE_8_BIT_QUANTIZATION
        /* 8-bit quantized multiplication */
    #if __ARM__
        INLINE void dot(const int8_t (&input)[INPUTS], float (&output)[OUTPUTS]) const
        {
            constexpr auto N = 16;  // NEON processes 16 bytes per register (vs 32 for AVX2)

            static_assert(INPUTS % N == 0);
            static_assert(OUTPUTS % N == 0);

            for (int j = 0; j != OUTPUTS; j += N)
            {
                // 16 partial sums, each is a 4x int32 vector
                int32x4_t partial_sum[N];

                #pragma unroll N
                for (int k = 0; k != N; ++k)
                    partial_sum[k] = vdupq_n_s32(0);

                for (int i = 0; i != INPUTS; i += N)
                {
                    // Load 16 uint8 inputs
                    uint8x16_t v_in = vld1q_u8(reinterpret_cast<const uint8_t*>(&input[i]));

                    for (int k = 0; k != N; ++k)
                    {
                        // Load 16 int8 weights
                        int8x16_t v_wt = vld1q_s8(&_wt[j + k][i]);

                        // Dot product: computes 4 dots of 4 elements each
                        // Assumes input in [0, 127] for correct unsigned×signed
                        partial_sum[k] = vdotq_s32(partial_sum[k], vreinterpretq_s8_u8(v_in), v_wt);
                    }
                }

                // Process outputs in groups of 4 (to vectorize the final stage)
                for (int n = 0; n < N; n += 4)
                {
                    // Horizontal add: reduce each partial_sum from 4x int32 to scalar
                    int32_t sums[4];
                    for (int i = 0; i < 4; ++i) {
                        int32x2_t sum_pairs = vadd_s32(
                            vget_low_s32(partial_sum[n+i]),
                            vget_high_s32(partial_sum[n+i])
                        );
                        sums[i] = vget_lane_s32(vpadd_s32(sum_pairs, sum_pairs), 0);
                    }

                    // Load as vector
                    int32x4_t i_out = vld1q_s32(sums);

                    // Load 4 biases (int8) and extend to int32
                    int8_t bias_bytes[4] = {_b[j + n], _b[j + n + 1], _b[j + n + 2], _b[j + n + 3]};
                    int16x4_t bias_i16 = vget_low_s16(vmovl_s8(vld1_s8(bias_bytes)));
                    int32x4_t bias_i32 = vmovl_s16(bias_i16);

                    // Multiply bias by QSCALE and add
                    int32x4_t qscale_vec = vdupq_n_s32(QSCALE_16);
                    bias_i32 = vmulq_s32(bias_i32, qscale_vec);
                    i_out = vaddq_s32(i_out, bias_i32);

                    // ReLU: max(i_out, 0)
                    i_out = vmaxq_s32(i_out, vdupq_n_s32(0));

                    // Convert to float
                    float32x4_t out = vcvtq_f32_s32(i_out);

                    // Divide by (QSCALE * QSCALE)
                    float32x4_t scale = vdupq_n_f32(float(QSCALE_8 * QSCALE_16));
                    out = vdivq_f32(out, scale);

                    // Store 4 outputs
                    ASSERT(j + n + 3 < OUTPUTS);
                    vst1q_f32(&output[j + n], out);
                }
            }
        }
    #else /* !__ARM__ */
        INLINE void dot(const int8_t (&input)[INPUTS], float (&output)[OUTPUTS]) const
        {
            constexpr auto N = Vec32c::size();

            static_assert(INPUTS % N == 0);
            static_assert(OUTPUTS % N == 0);

            Vec32uc v_in;
            Vec32c v_wt;
            Vec8i partial_sum[N];

            for (int j = 0; j != OUTPUTS; j += N)
            {
                #pragma unroll N
                for (int k = 0; k != N; ++k)
                    partial_sum[k] = 0;

                for (int i = 0; i != INPUTS; i += N)
                {
                    v_in.load_a(&input[i]);

                    for (int k = 0; k != N; ++k)
                    {
                        v_wt.load_a(&_wt[j + k][i]);
                        partial_sum[k] = mul_add(v_in, v_wt, partial_sum[k]);
                    }
                }

                #pragma unroll
                for (int n = 0; n < 32; n += 16)
                {
                    auto i_out = Vec16i(
                        horizontal_add(partial_sum[n+0]), horizontal_add(partial_sum[n+1]), horizontal_add(partial_sum[n+2]), horizontal_add(partial_sum[n+3]),
                        horizontal_add(partial_sum[n+4]), horizontal_add(partial_sum[n+5]), horizontal_add(partial_sum[n+6]), horizontal_add(partial_sum[n+7]),
                        horizontal_add(partial_sum[n+8]), horizontal_add(partial_sum[n+9]), horizontal_add(partial_sum[n+10]),horizontal_add(partial_sum[n+11]),
                        horizontal_add(partial_sum[n+12]),horizontal_add(partial_sum[n+13]),horizontal_add(partial_sum[n+14]),horizontal_add(partial_sum[n+15])
                    );

                    Vec16c bias;
                    bias.load_a(&_b[j + n]);
                    i_out += extend(extend(bias)) * QSCALE_16; // add converted 8bit -> 32bit bias
                    const Vec16f out = to_float(relu(i_out)) / float(QSCALE_8 * QSCALE_16);
                    ASSERT(j + n < OUTPUTS);
                    out.store_a(&output[j + n]);
                }
            }
        }
    #endif /* !__ARM__ */
    #endif /* USE_8_BIT_QUANTIZATION */

        /* hidden, output */
        template <typename ACTIVATION>
        static INLINE void dot(
            const float (&input)[INPUTS],
            float (&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS],
            ACTIVATION activate
        )
        {
            constexpr int N = Vector::size();
            constexpr int Q = (OUTPUTS % N == 0) ? N : OUTPUTS % N;

            static_assert(INPUTS % N == 0);
            static_assert(Q == N || Q == 1); /* result layer: Q == 1 */

            Vector sum[Q], v_wt, v_in, v_out;

            for (int j = 0; j != OUTPUTS; j += Q)
            {
                #pragma unroll Q
                for (int k = 0; k != Q; ++k)
                    sum[k] = Vector(0.0);

                #pragma unroll INPUTS
                for (int i = 0; i != INPUTS; i += N)
                {
                    v_in.load_a(&input[i]);

                    #pragma unroll Q
                    for (int k = 0; k != Q; ++k)
                    {
                        v_wt.load_a(&wt[j + k][i]);
                        sum[k] = mul_add(v_in, v_wt, sum[k]);
                    }
                }

                load_partial<Q>(v_out, &b[j]);
                v_out += horizontal_add(sum);
                store_partial<Q>(activate(v_out), &output[j]);
            }
        }

        template <size_t N, typename U, typename V>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS]) const
        {
            dot(input, output, _b, _wt, [](const Vector& v) { return v; });
        }

        template <size_t N, typename U, typename V, typename ACTIVATION>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS], ACTIVATION activate) const
        {
            dot(input, output, _b, _wt, activate);
        }
    };


    /** Get king bucket [index, mirror] */
    INLINE std::tuple<int, bool> get_king_bucket(int king_sq)
    {
        int file = king_sq % 8;
        int rank = king_sq / 8;
        const bool mirror = (file < 4);

        // Mirror and normalize file to 0-3
        file = mirror ? (3 - file) : (file - 4);
        ASSERT(file >= 0 && file <= 3);

        return std::make_tuple(rank * 4 + file, mirror);  // 0-31
    }


    template <Color COLOR>
    INLINE std::tuple<int, bool> get_king_bucket(const State& board)
    {
        const auto mask = board.occupied_co(COLOR);
        auto king_square = lsb(board.kings & mask);

        ASSERT(king_square == board.king(COLOR));

        // For black's perspective, mirror vertically
        if constexpr (COLOR == BLACK)
            king_square ^= 56;

        return get_king_bucket(king_square);
    }


    template <Color COLOR, typename T>
    INLINE void one_hot_encode(const State& board, int base, bool mirror, T (&perspective)[PERSPECTIVE_SIZE])
    {
        int i = 63;  // Start from bit 63, counting down
        const Bitboard masks[2] = {
            board._occupied_co[COLOR],   // friendly first
            board._occupied_co[!COLOR]   // enemy second
        };

        #pragma unroll 6
        for (const auto bb : {board.kings, board.pawns, board.knights, board.bishops, board.rooks, board.queens})
        {
            for (const auto mask : masks)
            {
                for_each_square_r((bb & mask), [&](int j) {
                    if constexpr (COLOR == BLACK)
                        j ^= 56;  // vertical mirror for black
                    if (mirror)
                        j ^= 7;   // horizontal mirror for bucketing
                    perspective[base + i - j] = 1;
                });
                i += 64;
            }
        }
    }


    template <int M, int N> struct Accumulator
    {
        static constexpr int INPUTS = M;
        static constexpr int OUTPUTS = N;

        using output_t = int16_t;

        uint64_t _hash = 0;

    #if DEBUG_INCREMENTAL
        ALIGN input_t _perspective[2][INPUTS] = {};
    #endif

        ALIGN output_t _output[2 * OUTPUTS] = {};

        INLINE output_t (&output(Color color))[OUTPUTS]
        {
            return *reinterpret_cast<output_t(*)[OUTPUTS]>(&_output[color * OUTPUTS]);
        }

        INLINE const output_t (&output(Color color) const)[OUTPUTS]
        {
            return *reinterpret_cast<const output_t(*)[OUTPUTS]>(&_output[color * OUTPUTS]);
        }

        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != _hash;
        }

        template <Color COLOR, typename L>
        INLINE void full_update(const L& layer, const State& state)
        {
            auto [king_bucket, mirror] = get_king_bucket<COLOR>(state);
        #if DEBUG_INCREMENTAL
            memset(&_perspective[COLOR], 0, sizeof(_perspective[COLOR]));
            auto& perspective = _perspective[COLOR];
        #else
            ALIGN input_t perspective[INPUTS] = {};
        #endif
            one_hot_encode<COLOR>(state, king_bucket * PIECE_FEATURES, mirror, perspective);
            memset(&output(COLOR), 0, sizeof(output(COLOR)));
            layer.dot(king_bucket, perspective, output(COLOR));
        }

        /** Fully update at root */
        template <typename L>
        INLINE void update(const L& black, const L& white, const State& state)
        {
            if (needs_update(state))
            {
                _hash = state.hash();

                full_update<BLACK>(black, state);
                full_update<WHITE>(white, state);
            }
        }

        template <Color COLOR, typename L, typename A>
        INLINE void update_perspective(
            const L& layer,
            const State& prev,
            const State& state,
            const BaseMove& move,
            A& ancestor)
        {
        #if __ARM__
            using VecShort = Vec16s;
        #else
            using VecShort = Vec32s;
        #endif /* __ARM__ */

            static_assert(L::OUTPUTS % VecShort::size() == 0);

            const auto [king_bucket, mirror] = get_king_bucket<COLOR>(state);
            const auto [prev_bucket, prev_mirror] = get_king_bucket<COLOR>(prev);

            if (king_bucket != prev_bucket || mirror != prev_mirror)
            {
            #if 0
                full_update<COLOR>(layer, state);
            #else
                // Start from ancestor's output
                memcpy(&output(COLOR)[0], &ancestor.output(COLOR)[0], sizeof(output(COLOR)));

                const int old_base = prev_bucket * PIECE_FEATURES;
                const int new_base = king_bucket * PIECE_FEATURES;

                VecShort vo, vw_old, vw_new;

                const Bitboard masks[2] = {
                    prev._occupied_co[COLOR],   // friendly first
                    prev._occupied_co[!COLOR]   // enemy second
                };

                // Iterate all pieces in prev position -- and shift into new bucket.
                int i = 63;
                for (const auto bb : {prev.kings, prev.pawns, prev.knights, prev.bishops, prev.rooks, prev.queens})
                {
                    for (const auto mask : masks)
                    {
                        for_each_square_r((bb & mask), [&](int sq) {
                            if constexpr (COLOR == BLACK)
                                sq ^= 56;

                            int old_sq = prev_mirror ? (sq ^ 7) : sq;
                            int new_sq = mirror ? (sq ^ 7) : sq;

                            int old_idx = old_base + i - old_sq;
                            int new_idx = new_base + i - new_sq;

                            if (old_idx != new_idx)
                            {
                                for (int j = 0; j < L::OUTPUTS; j += VecShort::size())
                                {
                                    vo.load_a(&output(COLOR)[j]);
                                    vw_old.load_a(&layer._w[old_idx][j]);
                                    vw_new.load_a(&layer._w[new_idx][j]);
                                    vo = vo - vw_old + vw_new;
                                    vo.store_a(&output(COLOR)[j]);
                                }
                            }
                        });
                        i += 64;
                    }
                }

                // Now apply move deltas on top
                if (move)
                {
                    int add[MAX_PIECES], remove[MAX_PIECES];
                    int add_count = 0, remove_count = 0;

                    compute_pieces_to_add_and_remove<COLOR>(prev, state, move, prev.turn, add, remove, add_count, remove_count, mirror);

                    for (int j = 0; j < L::OUTPUTS; j += VecShort::size())
                    {
                        vo.load_a(&output(COLOR)[j]);

                        for (int k = 0; k < remove_count; ++k)
                        {
                            vw_old.load_a(&layer._w[new_base + remove[k]][j]);
                            vo -= vw_old;
                        }
                        for (int k = 0; k < add_count; ++k)
                        {
                            vw_new.load_a(&layer._w[new_base + add[k]][j]);
                            vo += vw_new;
                        }

                        vo.store_a(&output(COLOR)[j]);
                    }
                }
            #if DEBUG_INCREMENTAL
                memset(&_perspective[COLOR], 0, sizeof(_perspective[COLOR]));
                one_hot_encode<COLOR>(state, new_base, mirror, _perspective[COLOR]);
            #endif /* DEBUG_INCREMENTAL */
            #endif /* 0 */
            }
            else
            {
                int add[MAX_PIECES], remove[MAX_PIECES];
                int add_count = 0, remove_count = 0;

                if (move)
                {
                    compute_pieces_to_add_and_remove<COLOR>(prev, state, move, prev.turn, add, remove, add_count, remove_count, mirror);

                    ASSERT(add_count < MAX_PIECES);
                    ASSERT(remove_count < MAX_PIECES);
                }
                const auto base = king_bucket * PIECE_FEATURES;

            #if DEBUG_INCREMENTAL
                /* Verify that compute_pieces_to_add_and_remove works correctly */
                memcpy(_perspective[COLOR], ancestor._perspective[COLOR], sizeof(_perspective[COLOR]));

                for (int i = 0; i != remove_count; ++i)
                    _perspective[COLOR][base + remove[i]] = 0;
                for (int i = 0; i != add_count; ++i)
                    _perspective[COLOR][base + add[i]] = 1;

                input_t temp[INPUTS] = {};
                one_hot_encode<COLOR>(state, base, mirror, temp);

                for (int i = 0; i != INPUTS; ++i)
                    ASSERT_ALWAYS(_perspective[COLOR][i] == temp[i]);
            #endif /* DEBUG_INCREMENTAL */

                /* apply incremental changes to previously computed output */
                memcpy(&output(COLOR)[0], &ancestor.output(COLOR)[0], sizeof(output(COLOR)));

                if (add_count || remove_count)
                {
                    VecShort vo, vw;
                    for (int j = 0; j != L::OUTPUTS; j += VecShort::size())
                    {
                        vo.load_a(&output(COLOR)[j]);
                        for (int i = 0; i < remove_count; ++i)
                        {
                            const auto index = base + remove[i];
                            ASSERT(index < L::INPUTS);
                            vw.load_a(&layer._w[index][j]);
                            vo -= vw;
                        }

                        for (int i = 0; i < add_count; ++i)
                        {
                            const auto index = base + add[i];
                            ASSERT(index < L::INPUTS);
                            vw.load_a(&layer._w[index][j]);
                            vo += vw;
                        }

                        vo.store_a(&output(COLOR)[j]);
                    }
                }

            #if DEBUG_INCREMENTAL
                /** Verify the incremental and full updates match */
                ALIGN output_t out[OUTPUTS] = { };
                layer.dot(king_bucket, _perspective[COLOR], out);
                for (int i = 0; i != OUTPUTS; ++i)
                    ASSERT_ALWAYS(abs(out[i] - output(COLOR)[i]) < 0.0001);
            #endif /* DEBUG_INCREMENTAL */
            }
        }

        /** Update incrementally, based on a previous state */
        template <typename L, typename A>
        INLINE void update(
            const L& black,
            const L& white,
            const State& prev,
            const State& state,
            const BaseMove& move,
            A& ancestor)
        {
            if (needs_update(state))
            {
                _hash = state.hash();

                /* compute delta based on ancestor state */
                ASSERT(prev.turn != state.turn);

                update_perspective<BLACK>(black, prev, state, move, ancestor);
                update_perspective<WHITE>(white, prev, state, move, ancestor);
            }
        }

        static INLINE constexpr int piece_square_index(PieceType piece_type, Color color, Square square)
        {
            ASSERT(piece_type);
            const auto index = (piece_type % 6) * 128 + (64 * color) + 63 - square;
            ASSERT(index < PIECE_FEATURES);
            return index;
        }

        template <Color PERSPECTIVE>
        static INLINE void delta(int (&d)[MAX_PIECES], int& idx, PieceType pt, Color col, Square sq, bool mirror)
        {
            ASSERT(sq != Square::UNDEFINED);
            int s = static_cast<int>(sq);
            if constexpr (PERSPECTIVE == BLACK)
                s ^= 56;  // vertical mirror for black's perspective
            if (mirror)
                s ^= 7;   // horizontal mirror for bucketing

            const auto us_or_them = (col == PERSPECTIVE) ? BLACK : WHITE;

            d[idx++] = piece_square_index(pt, us_or_them, Square(s));
        }

        template <Color PERSPECTIVE>
        INLINE void compute_pieces_to_add_and_remove(
            const State& from_pos,
            const State& to_pos,
            const Move& move,
            Color color, /* color of side that moved */
            int (&add)[MAX_PIECES],
            int (&remove)[MAX_PIECES],
            int& a_idx,
            int& r_idx,
            bool mirror)
        {
            ASSERT(move);

            if (const auto promo = move.promotion())
            {
                // add the promoted-to piece
                delta<PERSPECTIVE>(add, a_idx, promo, color, move.to_square(), mirror);

                // remove the pawn
                delta<PERSPECTIVE>(remove, r_idx, PieceType::PAWN, color, move.from_square(), mirror);
            }
            else
            {
                const auto ptype = from_pos.piece_type_at(move.from_square());

                delta<PERSPECTIVE>(remove, r_idx, ptype, color, move.from_square(), mirror);
                delta<PERSPECTIVE>(add, a_idx, ptype, color, move.to_square(), mirror);

                if (to_pos.is_castle)
                {
                    const auto king_file = square_file(move.to_square());
                    const auto rook_from_square = rook_castle_squares[king_file == 2][0][color];
                    const auto rook_to_square = rook_castle_squares[king_file == 2][1][color];

                    delta<PERSPECTIVE>(remove, r_idx, PieceType::ROOK, color, rook_from_square, mirror);
                    delta<PERSPECTIVE>(add, a_idx, PieceType::ROOK, color, rook_to_square, mirror);
                }
            }

            if (to_pos.is_capture())
            {
                const auto capture_square = from_pos.is_en_passant(move)
                    ? Square(from_pos.en_passant_square - 8 * SIGN[color])
                    : move.to_square();
                const auto victim_type = from_pos.piece_type_at(capture_square);

                delta<PERSPECTIVE>(remove, r_idx, victim_type, !color, capture_square, mirror);
            }
        }
    };


    template <typename A, typename L2, typename L3, typename EVAL>
    INLINE int eval(bool white_to_move, const A& a, const L2& l2, const L3& l3, const EVAL& eval)
    {
        // ProfileScope<class EVALUATE, 1000> profile;

    #if USE_8_BIT_QUANTIZATION
        ALIGN int8_t l2_in[L2::INPUTS];
    #else
        ALIGN float l2_in[L2::INPUTS];
    #endif

        ALIGN float l2_out[L2::OUTPUTS];
        ALIGN float l3_out[L3::OUTPUTS];
        ALIGN float output[1]; // eval

        using feature_t = typename A::output_t;
        feature_t features[2 * A::OUTPUTS];
        if (white_to_move)
        {
            /* swap perspectives, so that white comes first */
            memcpy(&features[0], a.output(WHITE), A::OUTPUTS * sizeof(feature_t));
            memcpy(&features[A::OUTPUTS], a.output(BLACK), A::OUTPUTS * sizeof(feature_t));

            activate(features, l2_in);
        }
        else
        {
            /* use accumulator directly (black perspective is first) */
            activate(a._output, l2_in);
        }

    #if USE_8_BIT_QUANTIZATION
        l2.dot(l2_in, l2_out);
    #else
        l2.dot(l2_in, l2_out, [](const Vector& v) { return max(v, v_zero); });
    #endif
        l3.dot(l2_out, l3_out,  [](const Vector& v) { return max(v, v_zero); });
        eval.dot(l3_out, output);
        return static_cast<int>(EVAL_SCALE * output[0]);
    }
} /* namespace nnue */
