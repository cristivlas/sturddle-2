#pragma once
/*
 * Sturddle Chess Engine (C) 2023 - 2025 Cristian Vlasceanu
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
    #else
        #define ARCH "SSE2"
    #endif /* INSTRSET*/
#endif /* ARCH */

#define ALIGN alignas(64)

#if INSTRSET >= 9 /* AVX 512 */
    constexpr int INPUT_STRIDE = 32;
#else
    constexpr int INPUT_STRIDE = 16;
#endif

#ifndef DEBUG_INCREMENTAL
    #define DEBUG_INCREMENTAL false
#endif

/* CXXFLAGS or CFLAGS="-march=armv8.2-a+fp16+dotprod" */
#if !__ARM__ || defined(__ARM_FEATURE_DOTPROD)
    #define USE_8_BIT_QUANTIZATION true
#endif

namespace nnue
{
    using namespace chess;
    using input_t = int16_t;

    constexpr int ACTIVE_INPUTS = 897;
    constexpr int CLIP_RELU = 127;
    constexpr int EVAL_SCALE = 16;
    constexpr int MAX_ACTIVE_INPUTS = 65; // 32 pieces + 32 occupancy mask + turn
    constexpr auto POOL_STRIDE = Vec8s::size();
    constexpr int QSCALE_16 = 255;
    constexpr int QSCALE_8 = 32;

    /* bit index of the side-to-move feature within one-hot encoding */
    constexpr int TURN_INDEX = 768;

    #if INSTRSET >= 9
        using Vector = Vec16f;

        INLINE Vector horizontal_add(const Vector (&v)[16])
        {
            return Vector(
                horizontal_add(v[0]), horizontal_add(v[1]), horizontal_add(v[2]), horizontal_add(v[3]),
                horizontal_add(v[4]), horizontal_add(v[5]), horizontal_add(v[6]), horizontal_add(v[7]),
                horizontal_add(v[8]), horizontal_add(v[9]), horizontal_add(v[10]),horizontal_add(v[11]),
                horizontal_add(v[12]),horizontal_add(v[13]),horizontal_add(v[14]),horizontal_add(v[15]));
        }
    #elif INSTRSET >= 7
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
    static const Vec8s  v8_zero(0);


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

    template <typename T>
    INLINE void one_hot_encode(const State& board, T (&encoding)[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)])
    {
        const auto& color_masks = board._occupied_co;
        int i = 63;

        #pragma unroll 6
        for (const auto bb : {board.kings, board.pawns, board.knights, board.bishops, board.rooks, board.queens})
        {
            #pragma unroll 2
            for (const auto mask : color_masks)
            {
                for_each_square_r((bb & mask), [&](Square j) { encoding[i - j] = 1; });
                i += 64;
            }
        }
        encoding[TURN_INDEX] = board.turn;

        for_each_square_r(color_masks[0], [&](Square j) { encoding[832 - j] = 1; });
        for_each_square_r(color_masks[1], [&](Square j) { encoding[896 - j] = 1; });
    }

    template <typename F>
    static INLINE void for_each_active_input(const State& state, F&& func)
    {
        const auto& color_masks = state._occupied_co;
        int i = 63;

        for (const auto bb : {state.kings, state.pawns, state.knights, state.bishops, state.rooks, state.queens})
        {
            for (const auto mask : color_masks)
            {
                for_each_square_r((bb & mask), [&](Square sq) { func(i - sq); });
                i += 64;
            }
        }

        if (state.turn)
            func(TURN_INDEX);

        // Occupancy masks
        for_each_square_r(color_masks[0], [&](Square sq) { func(832 - sq); });
        for_each_square_r(color_masks[1], [&](Square sq) { func(896 - sq); });
    }

    template <typename F>
    static INLINE void for_each_active_king_or_pawn(const State& state, F&& func)
    {
        const auto& color_masks = state._occupied_co;
        int i = 63;

        for (const auto bb : {state.kings, state.pawns})
        {
            for (const auto mask : color_masks)
            {
                for_each_square_r((bb & mask), [&](Square sq) { func(i - sq); });
                i += 64;
            }
        }
    }

    /** Calculate the piece-square index into the one-hot encoding. */
    INLINE constexpr int piece_square_index(PieceType piece_type, Color color, Square square)
    {
        return (piece_type % 6) * 128 + (64 * color) + 63 - square;
    }

    /** Calculate index into occupancy mask for given color */
    INLINE constexpr int mask_index(Color color, Square square) noexcept
    {
        return (color ? 833 : 769) + 63 - square;
    }


    /** Rectified Linear Unit (reLU) activation */
    template <typename V> INLINE V relu(V v) { return max(v, 0); }
    template <typename V> INLINE V clipped_relu(V v) { return min(CLIP_RELU, relu(v)); }

    template <>
    INLINE Vector relu<Vector>(Vector v) { return max(v, v_zero); }

    template <>
    INLINE Vec8s relu<Vec8s>(Vec8s v) { return max(v, v8_zero); }

#if !USE_8_BIT_QUANTIZATION
    template <int N>
    INLINE void activate_attn(const int16_t (&input)[N], float (&output)[N])
    {
        constexpr float QSCALE_RECIP = 1.0f / QSCALE_16;

#if __ARM__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        /* Vec8f supported only on FP16 (half-precision) Neon */
        #pragma clang loop vectorize(enable)
        for (int i = 0; i != N; ++i)
            output[i] = std::max<float>(0, float(std::min<int>(CLIP_RELU, input[i])) * QSCALE_RECIP);
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
    INLINE void activate_attn(const int16_t (&input)[N], int8_t (&output)[N])
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
    INLINE void activate_attn(const int16_t (&input)[N], int8_t (&output)[N])
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



#if INSTRSET >= 9 /* AVX-512 */
    /* Overflow is prevented at training time */
    INLINE Vec32s mul_add(Vec32s a, Vec32s b, Vec32s acc)
    {
        return acc + a * b;
    }
#elif __ARM__
    INLINE Vec16s mul_add(Vec16s a, Vec16s b, Vec16s acc)
    {
        return acc + a * b;
    }
#else
    INLINE Vec8i mul_add(Vec16s a, Vec16s b, Vec8i acc)
    {
    #if __AVXVNNI
        return _mm256_dpwssd_epi32(acc, a, b);
    #elif INSTRSET < 8
        /* SSE2 */
        // Multiply a * b and accumulate neighbouring outputs into int32 values
        __m128i product_lo = _mm_madd_epi16(a.get_low(), b.get_low());
        __m128i product_hi = _mm_madd_epi16(a.get_high(), b.get_high());
        // Add to the main int32 accumulator
        return Vec8i(_mm_add_epi32(acc.get_low(), product_lo), _mm_add_epi32(acc.get_high(), product_hi));
    #else
        /* AVX2 */
        __m256i product = _mm256_madd_epi16(a, b);
        return _mm256_add_epi32(acc, product);
    #endif
    }
#endif /* INSTRSET >= 9 */


    template <int I, int O, typename T, int Scale, bool Incremental>
    struct BaseLayer
    {
        static constexpr int ROWS = I;
        static constexpr int COLS = O;
        /* Round up to INPUT_STRIDE to deal with odd inputs. */
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
                    _b[j] = std::round(b[j] * Scale);

            for (int i = 0; i != I; ++i)
            {
                for (int j = 0; j != OUTPUTS; ++j)
                {
                    if constexpr (Scale == 1)
                        _wt[j][i] = w[i][j];
                    else
                        _wt[j][i] = std::round(w[i][j] * Scale);

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

        /* input */
        template <size_t INPUT_SIZE>
        INLINE void dot(const input_t (&input)[INPUT_SIZE], int16_t (&output)[OUTPUTS], size_t base = 0) const
        {
        #if INSTRSET >= 9 /* AVX 512 */
            using VecShort = Vec32s;
            using VSum = Vec32s;
        #elif __ARM__
            using VecShort = Vec16s;
            using VSum = Vec16s;
        #else
            using VecShort = Vec16s;
            using VSum = Vec8i;
        #endif /* INSTRSET */

            constexpr auto N = VecShort::size();
            static_assert(N == INPUT_STRIDE);
            static_assert(OUTPUTS % N == 0);

            VecShort in, vw;
            VSum sum[N]; /* accumulate partial sums */

            constexpr auto INPUT_MAX = std::min<int>(INPUTS, INPUT_SIZE);

            for (int j = 0; j != OUTPUTS; j += N)
            {
                #pragma unroll N
                for (int k = 0; k != N; ++k)
                    sum[k] = VSum(0);

                for (int i = 0; i < INPUT_MAX; i += N)
                {
                    in.load_a(input + i);
                    if (all_zero(in))
                        continue;

                    for (int k = 0; k != N; ++k)
                    {
                        vw.load(&_wt[j + k][i + base]);
                        sum[k] = mul_add(in, vw, sum[k]);
                    }
                }

                const auto sums = horizontal_add(sum);
                vw.load_a(&_b[j]);
                (vw + sums).store_a(&output[j]);
            }
        }

  #if USE_8_BIT_QUANTIZATION
        /* 8-bit quantized multiplication */
        /* NOTE: Linear (no activation) */
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
                    // i_out = vmaxq_s32(i_out, vdupq_n_s32(0));

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

                    //const Vec16f out = to_float(relu(i_out)) / float(QSCALE_8 * QSCALE_16);
                    const Vec16f out = to_float(i_out) / float(QSCALE_8 * QSCALE_16);

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


    template <size_t INPUTS, size_t OUTPUTS>
    INLINE void pool(const int16_t (&in)[INPUTS], float (&out)[OUTPUTS])
    {
        static_assert(INPUTS % OUTPUTS == 0);
        static_assert(INPUTS / OUTPUTS == POOL_STRIDE);

    #if INSTRSET < 8
        Vec8s v;

        for (size_t i = 0, j = 0; i + POOL_STRIDE <= INPUTS; i += POOL_STRIDE, ++j)
        {
            v.load_a(&in[i]);
            v = max(v, v8_zero);

            out[j] = float(::horizontal_add(extend(v))) / POOL_STRIDE / QSCALE_16;
        }
    #else
        /* AVX2 (or better) */
        static_assert(INPUTS % (2 * POOL_STRIDE) == 0);
        Vec16s v;

        for (size_t i = 0, j = 0; i + 2 * POOL_STRIDE <= INPUTS; i += 2 * POOL_STRIDE, j += 2)
        {
            v.load_a(&in[i]);

            ASSERT(j < OUTPUTS);
            ASSERT(j + 1< OUTPUTS);

            out[j] = float(::horizontal_add(extend(max(v.get_low(), v8_zero)))) / POOL_STRIDE / QSCALE_16;
            out[j + 1] = float(::horizontal_add(extend(max(v.get_high(), v8_zero)))) / POOL_STRIDE / QSCALE_16;
        }
    #endif /* INSTRSET < 8 */
    }


    template <int M, int N, int O> struct Accumulator
    {
        static_assert(ACTIVE_INPUTS * 4 == M);

        static constexpr int INPUTS = round_up<INPUT_STRIDE>(M);
        static constexpr int OUTPUTS_A = N;
        static constexpr int OUTPUTS_B = O;
        static constexpr int NUM_BUCKETS = 4;

        struct Bucket
        {
            ALIGN int16_t output[OUTPUTS_A] = { };
            uint64_t hash = 0;
        };

        Bucket _bucket[NUM_BUCKETS];
        int _current_bucket = 0;
        ALIGN int16_t _output_b[OUTPUTS_B] = { };

    #if DEBUG_INCREMENTAL
        /* remember previous inputs, for debugging */
        ALIGN input_t _input[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)] = { }; /* one-hot encoding */
    #endif


        static INLINE int get_bucket(const State& state)
        {
            return std::min<int>(chess::popcount(state.pawns) / 4, 3);
        }


        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != _bucket[_current_bucket].hash;
        }


        /** Compute 1st layer output from scratch at root */
        template <typename LA, typename LB>
        INLINE void full_update(const LA& layer_1a, const LB& layer_1b, const State& state, int bucket)
        {
            const size_t base = bucket * ACTIVE_INPUTS;

        #if DEBUG_INCREMENTAL
            memset(&_input, 0, sizeof(_input));
        #else
            ALIGN input_t _input[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)] = { };
        #endif
            one_hot_encode(state, _input);

            layer_1a.dot(_input, _bucket[bucket].output, base);
            layer_1b.dot(_input, _output_b);

            _bucket[bucket].hash = state.hash();
            _current_bucket = bucket;
        }

        template <typename LA, typename LB>
        INLINE void update(const LA& layer_1a, const LB& layer_1b, const State& state)
        {
            if (needs_update(state))
            {
                full_update(layer_1a, layer_1b, state, get_bucket(state));
            }
        }

        /** Utility for incremental updates */
        static INLINE void delta(int (&d)[MAX_ACTIVE_INPUTS], int& idx, PieceType pt, Color col, Square sq)
        {
            d[idx++] = piece_square_index(pt, col, sq);
            d[idx++] = mask_index(col, sq);
        }

        /** Update 1st layer output incrementally, based on a previous state */
        template <typename LA, typename LB>
        INLINE void update(
            const LA& layer_a,
            const LB& layer_b,
            const State& prev,
            const State& state,
            const Move& move,
            Accumulator& ancestor)
        {
            ASSERT(needs_update(state));
            ASSERT(ancestor._bucket[ancestor._current_bucket].hash == prev.hash());

            const int bucket = get_bucket(state);
            bool incremental_a = (ancestor._bucket[bucket].hash == prev.hash());

            /* compute delta based on ancestor state */
            ASSERT(prev.turn != state.turn);

            int remove_inputs[MAX_ACTIVE_INPUTS];
            int add_inputs[MAX_ACTIVE_INPUTS];
            int r_idx = 0, a_idx = 0;

            if (move)
            {
                get_deltas(prev, state, move, prev.turn, remove_inputs, add_inputs, r_idx, a_idx);

                ASSERT(a_idx < MAX_ACTIVE_INPUTS);
                ASSERT(r_idx < MAX_ACTIVE_INPUTS);
            }

        #if DEBUG_INCREMENTAL
            memcpy(_input, ancestor._input, sizeof(_input));

            // Validate get_deltas
            for (int i = 0; i != r_idx; ++i)
                _input[remove_inputs[i]] = 0;
            for (int i = 0; i != a_idx; ++i)
                _input[add_inputs[i]] = 1;

            _input[TURN_INDEX] ^= 1;

            ALIGN input_t temp[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)] = { };
            one_hot_encode(state, temp);

            for (int i = 0; i != ACTIVE_INPUTS; ++i)
                ASSERT_ALWAYS(_input[i] == temp[i]);
        #endif /* DEBUG_INCREMENTAL */

            if (state.turn)
                add_inputs[a_idx++] = TURN_INDEX;
            else
                remove_inputs[r_idx++] = TURN_INDEX;

            const size_t base = bucket * ACTIVE_INPUTS;

            if (incremental_a)
            {
                memcpy(_bucket[bucket].output, ancestor._bucket[bucket].output, sizeof(_bucket[bucket].output));
            }
            else if (_bucket[bucket].hash != state.hash())
            {
                const int prev_bucket = ancestor._current_bucket;
                const size_t base_old = prev_bucket * ACTIVE_INPUTS;
                const size_t base_new = bucket * ACTIVE_INPUTS;

                // Start from ancestor's output (computed with prev_bucket weights for prev position)
                memcpy(_bucket[bucket].output, ancestor._bucket[prev_bucket].output, sizeof(_bucket[bucket].output));

            #if __ARM__
                using VecShort = Vec16s;
            #else
                using VecShort = Vec32s;
            #endif
                static_assert(OUTPUTS_A % VecShort::size() == 0);

                auto apply_delta = [&](int idx)
                {
                    VecShort vo, vw_old, vw_new;

                    for (int j = 0; j < OUTPUTS_A; j += VecShort::size())
                    {
                        vo.load_a(&_bucket[bucket].output[j]);
                        vw_old.load_a(&layer_a._w[base_old + idx][j]);
                        vw_new.load_a(&layer_a._w[base_new + idx][j]);
                        vo = vo - vw_old + vw_new;
                        vo.store_a(&_bucket[bucket].output[j]);
                    }
                };
                for_each_active_input(prev, apply_delta);

                // Now _bucket[bucket].output has correct output for prev position with bucket weights
                // Set flag so move deltas get applied
                incremental_a = true;
            }

            /* layer B: update incrementally from ancestor state */
            memcpy(_output_b, ancestor._output_b, sizeof(_output_b));
            incremental_update(layer_a, layer_b, remove_inputs, add_inputs, r_idx, a_idx, base, bucket, incremental_a);

            _bucket[bucket].hash = state.hash();
            _current_bucket = bucket;

        #if DEBUG_INCREMENTAL
            // Validate that incremental_update produces same result as full dot products
            // layer A
            ALIGN int16_t output_a[OUTPUTS_A] = { };
            layer_a.dot(temp, output_a, base);
            for (int i = 0; i != OUTPUTS_A; ++i)
                ASSERT_ALWAYS(abs(output_a[i] - _bucket[bucket].output[i]) < 0.0001);

            // layer B
            ALIGN int16_t output_b[OUTPUTS_B] = { };
            layer_b.dot(temp, output_b);
            for (int i = 0; i != OUTPUTS_B; ++i)
                ASSERT_ALWAYS(abs(output_b[i] - _output_b[i]) < 0.0001);
        #endif /* DEBUG_INCREMENTAL */
        }

        /** Recompute incrementally */
        template <typename LA, typename LB>
        INLINE void incremental_update(
            const LA& layer_a,
            const LB& layer_b,
            const int (&remove_inputs)[MAX_ACTIVE_INPUTS],
            const int (&add_inputs)[MAX_ACTIVE_INPUTS],
            const int r_idx,
            const int a_idx,
            size_t base,
            int bucket,
            bool update_layer_a)
        {
        #if __ARM__
            using VecShort = Vec16s;
        #else
            using VecShort = Vec32s;
        #endif /* __ARM__ */

            static_assert(LA::OUTPUTS == OUTPUTS_A);
            static_assert(LB::OUTPUTS == OUTPUTS_B);
            static_assert(LA::OUTPUTS % VecShort::size() == 0);
            static_assert(LB::OUTPUTS % VecShort::size() == 0);

            int update_layer_b = 0;
            for (int i = 0; i < r_idx && !update_layer_b; ++i)
                update_layer_b += remove_inputs[i] < LB::INPUTS;
            for (int i = 0; i < a_idx && !update_layer_b; ++i)
                update_layer_b += add_inputs[i] < LB::INPUTS;

            VecShort vo, vw;

            /* Layer A */
            if (update_layer_a)
            {
                for (int j = 0; j != OUTPUTS_A; j += VecShort::size())
                {
                    vo.load_a(&_bucket[bucket].output[j]);

                    for (int i = 0; i < r_idx; ++i)
                    {
                        const auto index = base + remove_inputs[i];
                        ASSERT(index < LA::INPUTS);
                        vw.load_a(&layer_a._w[index][j]);
                        vo -= vw;
                    }

                    for (int i = 0; i < a_idx; ++i)
                    {
                        const auto index = base + add_inputs[i];
                        ASSERT(index < LA::INPUTS);
                        vw.load_a(&layer_a._w[index][j]);
                        vo += vw;
                    }
                    vo.store_a(&_bucket[bucket].output[j]);
                }
            }

            if (update_layer_b)
            {
                /* Layer B */
                for (int j = 0; j != OUTPUTS_B; j += VecShort::size())
                {
                    vo.load_a(&_output_b[j]);

                    for (int i = 0; i < r_idx; ++i)
                    {
                        const auto index = remove_inputs[i];
                        if (index >= LB::INPUTS)
                            continue;
                        vw.load_a(&layer_b._w[index][j]);
                        vo -= vw;
                    }

                    for (int i = 0; i < a_idx; ++i)
                    {
                        const auto index = add_inputs[i];
                        if (index >= LB::INPUTS)
                            break;
                        vw.load_a(&layer_b._w[index][j]);
                        vo += vw;
                    }
                    vo.store_a(&_output_b[j]);
                }
            }
        }

        /** Get the indices of pieces to add / remove */
        INLINE void get_deltas(
            const State& from_pos,
            const State& to_pos,
            const Move& move,
            Color color, /* color of side that moved */
            int (&remove)[MAX_ACTIVE_INPUTS],
            int (&add)[MAX_ACTIVE_INPUTS],
            int& r_idx,
            int& a_idx)
        {
            if (const auto promo = move.promotion())
            {
                // add the promoted-to piece
                delta(add, a_idx, promo, color, move.to_square());

                // remove the pawn
                delta(remove, r_idx, PieceType::PAWN, color, move.from_square());
            }
            else
            {
                const auto ptype = from_pos.piece_type_at(move.from_square());

                delta(remove, r_idx, ptype, color, move.from_square());
                delta(add, a_idx, ptype, color, move.to_square());

                if (to_pos.is_castle)
                {
                    const auto king_file = square_file(move.to_square());
                    const auto rook_from_square = rook_castle_squares[king_file == 2][0][color];
                    const auto rook_to_square = rook_castle_squares[king_file == 2][1][color];

                    delta(remove, r_idx, PieceType::ROOK, color, rook_from_square);
                    delta(add, a_idx, PieceType::ROOK, color, rook_to_square);
                }
            }

            if (to_pos.is_capture())
            {
                const auto capture_square = from_pos.is_en_passant(move)
                    ? Square(from_pos.en_passant_square - 8 * SIGN[color])
                    : move.to_square();
                const auto victim_type = from_pos.piece_type_at(capture_square);

                delta(remove, r_idx, victim_type, !color, capture_square);
            }
        }
    };


    template <typename A, typename ATTN, typename L2, typename L3, typename OUT>
    INLINE int eval(const A& a, const ATTN& attn, const L2& l2, const L3& l3, const OUT& out)
    {
        static_assert(A::OUTPUTS_A == L2::INPUTS * POOL_STRIDE);
        static_assert(A::OUTPUTS_B == ATTN::INPUTS);

    #if USE_8_BIT_QUANTIZATION
        ALIGN int8_t attn_in[ATTN::INPUTS];
    #else
        ALIGN float attn_in[ATTN::INPUTS];
    #endif /* USE_8_BIT_QUANTIZATION */
        ALIGN float attn_out[ATTN::OUTPUTS];
        ALIGN float l2_in[L2::INPUTS];
        ALIGN float l2_out[L2::OUTPUTS];
        ALIGN float l3_out[L3::OUTPUTS];
        ALIGN float eval[1];

        pool(a._bucket[a._current_bucket].output, l2_in);

        /* The "spatial attention" layer modulates L2. */
        activate_attn(a._output_b, attn_in);
        attn.dot(attn_in, attn_out);

        static_assert(L2::INPUTS % Vector::size() == 0);

        Vector v1, v2;
        for (int i = 0; i != L2::INPUTS; i += Vector::size())
        {
            v1.load_a(&l2_in[i]);
            v2.load_a(&attn_out[i % ATTN::OUTPUTS]);

            mul_add(v1, v2, v1).store_a(&l2_in[i]);
        }
        /* end of modulation */

        l2.dot(l2_in, l2_out, [](const Vector& v) { return max(v, v_zero); });
        l3.dot(l2_out, l3_out, [](const Vector& v) { return max(v, v_zero); });

        out.dot(l3_out, eval);
        return EVAL_SCALE * eval[0];
    }


    template <typename LM>
    INLINE void score_move(const LM& layer_m, const int (&active)[MAX_ACTIVE_INPUTS], int count, Move& move)
    {
        const auto index = move.from_square() * 64 + move.to_square();

        // Start with bias
        auto score = layer_m._b[index];

        // Add contribution from each active feature
        for (int k = 0; k < count; ++k)
        {
            score += layer_m._wt[index][active[k]];
        }

        using move_score_t = decltype(move._score);
        move._score = std::min(std::numeric_limits<move_score_t>::max(), score);
    }
} /* namespace nnue */
