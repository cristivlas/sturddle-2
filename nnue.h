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
#include <istream>

#if (__amd64__) || (__x86_64__) || (__i386__) || (_M_AMD64) || (_M_X64) || (_M_IX86)
    #include "vectorclass.h"
    #if defined(__AVX512BF16__) && defined(__AVX512VL__)
        #define USE_BF16 true
    #else
        #define USE_BF16 false
    #endif
#elif (__arm__) || (__arm64__) || (__aarch64__)
    #define __ARM__ true
    #include "armvector.h"
#endif

#if __AVXVNNI__ || __AVX512VNNI__
    #define ARCH_VNNI "/VNNI"
#else
    #define ARCH_VNNI
#endif /* __AVXVNNI__ */

#ifdef __FMA__  /* support fused multiply+add? */
    #define ARCH_FMA "/FMA"
#else
    #define ARCH_FMA
#endif /* __FMA__ */

#if USE_BF16
    #define ARCH_BF16 "/BF16"
#else
    #define ARCH_BF16
#endif /* USE_BF16 */

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

#ifndef NNUE_SINGLE_BUCKET
    #define NNUE_SINGLE_BUCKET true
#endif

namespace nnue
{
    static const std::string instrset = ARCH ARCH_FMA ARCH_VNNI ARCH_BF16;

    using namespace chess;
    using input_t = int16_t;
    using weight_t = float;

    constexpr int ACTIVE_INPUTS = 769;
    constexpr int EVAL_SCALE = 100;
    constexpr int MAX_ACTIVE_INPUTS = 33; // 32 pieces + turn
    constexpr int NUM_BUCKETS = 16;
    constexpr int PAWN_BUCKETS = chess::PAWN_BUCKETS;
    constexpr int KING_BUCKETS = 4;
    static_assert(NUM_BUCKETS == PAWN_BUCKETS * KING_BUCKETS, "bucket grid mismatch");
    constexpr int POOL_STRIDE = 8;
    constexpr int QSCALE = 1024;
    constexpr int QLOG2 = 10;  /* log2(QSCALE), for shift-based requantization */
    static_assert((1 << QLOG2) == QSCALE, "QLOG2 must be log2(QSCALE)");

    /* bit index of the side-to-move feature within one-hot encoding */
    constexpr int TURN_INDEX = 768;

#if ATTACK_MASKS
    /* threat planes feeding hidden_1c: (king,pawn,knight,bishop,rook,queen) x (black,white) */
    constexpr int THREAT_PLANES = 12;
    constexpr int THREAT_INPUTS = THREAT_PLANES * 64;
    constexpr int THREATS_OUT = 32; /* hidden_1c width, must match the trained net */
#endif /* ATTACK_MASKS */

    INLINE int pawn_bucket(const State& state)
    {
        return chess::pawn_bucket(state.pawns);
    }

    INLINE int king_bucket(const State& state)
    {
        const int wk_right = square_file(state.king(WHITE)) >= 4;
        const int bk_right = square_file(state.king(BLACK)) >= 4;
        return wk_right * 2 + bk_right;
    }

    INLINE int get_bucket(const State& state)
    {
        return pawn_bucket(state) * KING_BUCKETS + king_bucket(state);
    }

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


    static const Vector v_zero(0.0);
    static const Vec8s  v8_zero(0);

    /* Type trait: selects vector type based on weight storage type */
    template<typename T> struct Vec { using type = Vector; };

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

    template<> struct Vec<__bf16> { using type = Vec32_bf16; };

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

    template <typename V>
    INLINE Vec32s horizontal_add(const V (&v)[32])
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


    /** Rectified Linear Unit (reLU) activation */
    template <typename V> INLINE V relu(V v) { return max(v, 0); }

    template <>
    INLINE Vector relu<Vector>(Vector v) { return max(v, v_zero); }

    template <>
    INLINE Vec8s relu<Vec8s>(Vec8s v) { return max(v, v8_zero); }


#if 0
    template <int N>
    INLINE void activate(const int16_t (&input)[N], float (&output)[N])
    {
        constexpr float QSCALE_RECIP = 1.0f / QSCALE;

#if __ARM__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        /* Vec8f supported only on FP16 (half-precision) Neon */
        #pragma clang loop vectorize(enable)
        for (int i = 0; i != N; ++i)
            output[i] = std::max<float>(0, float(input[i]) * QSCALE_RECIP);
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
            VF v = to_float(extend(relu(VS().load_a(&input[i]))));
            (v * v_scale).store_a(&output[i]);
        }
#endif /* __ARM__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    }
#endif /* 0 */


    /** ReLU + dequantize int32/QSCALE sums to float. */
    template <int N>
    INLINE void activate(const int32_t (&input)[N], float (&output)[N])
    {
        constexpr float QSCALE_RECIP = 1.0f / QSCALE;

#if __ARM__
        /* Vec8f32, not the fp16 Vec8f: the unbounded relu needs full float precision */
        static_assert(N % Vec8i::size() == 0);
        const Vec8i vi_zero(0);
        for (int i = 0; i != N; i += Vec8i::size())
        {
            Vec8i v;
            v.load_a(&input[i]);
            (Vec8f32(max(v, vi_zero)) * QSCALE_RECIP).store_a(&output[i]);
        }
#else
    #if INSTRSET < 9
        using VF = Vec8f;
        using VI = Vec8i;
    #else
        using VF = Vec16f;
        using VI = Vec16i;
    #endif /* AVX512 */
        static_assert(N % VF::size() == 0);
        const VF v_scale(QSCALE_RECIP);
        const VI v_izero(0);
        for (int i = 0; i != N; i += VF::size())
        {
            VI v;
            v.load_a(&input[i]);
            (to_float(max(v, v_izero)) * v_scale).store_a(&output[i]);
        }
#endif /* __ARM__ */
    }


    /** Dequantize int16/QSCALE to float, no activation (linear layer). */
    template <int N>
    INLINE void dequantize(const int16_t (&input)[N], float (&output)[N])
    {
        constexpr float QSCALE_RECIP = 1.0f / QSCALE;

#if __ARM__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        #pragma clang loop vectorize(enable)
        for (int i = 0; i != N; ++i)
            output[i] = float(input[i]) * QSCALE_RECIP;
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
            VF v = to_float(extend(VS().load_a(&input[i])));
            (v * v_scale).store_a(&output[i]);
        }
#endif /* __ARM__ && !__ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    }


#if INSTRSET >= 9 /* AVX-512 */
    #if __AVX512VNNI__
        INLINE Vec16i mul_add(Vec32s a, Vec32s b, Vec16i acc)
        {
            // VPDPWSSD: dot-product of signed 16-bit pairs into signed 32-bit dwords
            return Vec16i(_mm512_dpwssd_epi32(acc, a, b));
        }
    #else
        INLINE Vec16i mul_add(Vec32s a, Vec32s b, Vec16i acc)
        {
            __m512i product = _mm512_madd_epi16(a, b);
            return _mm512_add_epi32(acc, product);
        }
    #endif /* __AVX512VNNI__ */
#elif __ARM__
    INLINE Vec16s mul_add(Vec16s a, Vec16s b, Vec16s acc)
    {
        return acc + a * b;
    }
#else
    INLINE Vec8i mul_add(Vec16s a, Vec16s b, Vec8i acc)
    {
    #if __AVXVNNI__
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


    template <int I, int O, typename T, int Scale, bool Incremental, bool Transposed>
    struct BaseLayer
    {
        static_assert(Incremental && Transposed, "unsupported storage combination");

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
    struct BaseLayer<I, O, T, Scale, false, true>
    {
        static constexpr int ROWS = I;
        static constexpr int COLS = O;
        static constexpr int INPUTS = (Scale == 1) ? I : round_up<INPUT_STRIDE>(I);
        static constexpr int OUTPUTS = O;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */
    };


    /* Row-major weights only, for sparse input-major gather (hidden_1c) */
    template <int I, int O, typename T, int Scale>
    struct BaseLayer<I, O, T, Scale, true, false>
    {
        static constexpr int ROWS = I;
        static constexpr int COLS = O;
        static constexpr int INPUTS = round_up<INPUT_STRIDE>(I);
        static constexpr int OUTPUTS = O;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _w[INPUTS][OUTPUTS]; /* weights */
    };


    template <int I, int O, typename T=weight_t, int Scale=1, bool Incremental=false, bool Transposed=true>
    struct Layer : BaseLayer<I, O, T, Scale, Incremental, Transposed>
    {
        using Base = BaseLayer<I, O, T, Scale, Incremental, Transposed>;
        using Base::INPUTS;
        using Base::OUTPUTS;
        using Base::_b;

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
                    T v;
                    if constexpr (Scale == 1)
                        v = w[i][j];
                    else
                        v = std::round(w[i][j] * Scale);

                    if constexpr (Transposed)
                        this->_wt[j][i] = v;
                    if constexpr (Incremental)
                        this->_w[i][j] = v;
                }
            }
            /* padding, if needed */
            for (int i = I; i != INPUTS; ++i)
            {
                for (int j = 0; j != OUTPUTS; ++j)
                {
                    if constexpr (Transposed)
                        this->_wt[j][i] = 0;
                    if constexpr (Incremental)
                        this->_w[i][j] = 0;
                }
            }
        }

        void load_weights(std::istream& file)
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
            using VSum = Vec16i;
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
            static_assert(INPUT_SIZE % N == 0);

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
                        vw.load(&this->_wt[j + k][i + base]);
                        sum[k] = mul_add(in, vw, sum[k]);
                    }
                }

                const auto sums = horizontal_add(sum);
                static_assert(sums.size() == N);
                vw.load_a(&_b[j]);
                (vw + sums).store_a(&output[j]);
            }
        }

        /* hidden, output */
        template <size_t INPUT_SIZE, typename WT, typename ACTIVATION>
        static INLINE void dot(
            const float (&input)[INPUT_SIZE],
            float (&output)[OUTPUTS],
            const WT(&b)[OUTPUTS],
            const WT(&wt)[OUTPUTS][INPUTS],
            ACTIVATION activate,
            size_t base = 0
        )
        {
            constexpr int N = Vector::size();
            constexpr int Q = (OUTPUTS % N == 0) ? N : OUTPUTS % N;

            static_assert(INPUT_SIZE % N == 0);
            static_assert(Q == N || Q == 1); /* result layer: Q == 1 */

            Vector sum[Q], v_out;
            typename Vec<WT>::type v_in, v_wt;
            static_assert(INPUT_SIZE % v_in.size() == 0);

            for (int j = 0; j != OUTPUTS; j += Q)
            {
                #pragma unroll Q
                for (int k = 0; k != Q; ++k)
                    sum[k] = Vector(0.0);

                #pragma unroll INPUT_SIZE
                for (size_t i = 0; i != INPUT_SIZE; i += v_in.size())
                {
                    v_in.load_a(&input[i]);

                    #pragma unroll Q
                    for (int k = 0; k != Q; ++k)
                    {
                        v_wt.load_a(&wt[j + k][base + i]);
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
            dot(input, output, _b, this->_wt, [](const Vector& v) { return v; }, 0);
        }

        template <size_t N, typename U, typename V>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS], size_t base) const
        {
            dot(input, output, _b, this->_wt, [](const Vector& v) { return v; }, base);
        }

        template <size_t N, typename U, typename V, typename ACTIVATION>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS], ACTIVATION activate) const
        {
            dot(input, output, _b, this->_wt, activate, 0);
        }

        template <size_t N, typename U, typename V, typename ACTIVATION>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS], ACTIVATION activate, size_t base) const
        {
            dot(input, output, _b, this->_wt, activate, base);
        }
    };


    /* Learned pooling weights, one set per side to move; defaults to average pooling.
     * Weights are stored pre-multiplied by 1/QSCALE (exact, power of two), so pool()
     * dequantizes for free -- in-memory values differ from the file by that factor.
     */
    template <int N>
    struct PoolLayer
    {
        ALIGN float _w[2][N];

        PoolLayer()
        {
            for (auto& w : _w)
                std::fill(std::begin(w), std::end(w), 1.0f / POOL_STRIDE / QSCALE);
        }

        static constexpr size_t param_count() { return 2 * N; }

        void load_weights(std::istream& file)
        {
            file.read(reinterpret_cast<char*>(_w), sizeof(_w));

            for (auto& w : _w)
                for (auto& v : w)
                    v /= QSCALE;
        }
    };

    template <size_t INPUTS, size_t OUTPUTS>
    INLINE void pool(const int16_t (&in)[INPUTS], const float (&w)[INPUTS], float (&out)[OUTPUTS])
    {
        static_assert(INPUTS % OUTPUTS == 0);
        static_assert(INPUTS / OUTPUTS == POOL_STRIDE);
        static_assert(POOL_STRIDE == 8);

        /* w carries the 1/QSCALE factor, see PoolLayer */
#if __ARM__
        for (size_t i = 0, j = 0; i + POOL_STRIDE <= INPUTS; i += POOL_STRIDE, ++j)
        {
            float sum = 0;
            #pragma clang loop vectorize(enable)
            for (int k = 0; k != POOL_STRIDE; ++k)
                sum += float(std::max<int16_t>(0, in[i + k])) * w[i + k];
            out[j] = sum;
        }
#else
        Vec8s v;
        Vec8f vw;
        for (size_t i = 0, j = 0; i + POOL_STRIDE <= INPUTS; i += POOL_STRIDE, ++j)
        {
            v.load_a(&in[i]);
            vw.load_a(&w[i]);
            ASSERT(j < OUTPUTS);
            out[j] = ::horizontal_add(to_float(extend(max(v, v8_zero))) * vw);
        }
#endif /* __ARM__ */
    }


    template <int M, int N, int O> struct Accumulator
    {
        static_assert(ACTIVE_INPUTS * NUM_BUCKETS == M);

        static constexpr int INPUTS = round_up<INPUT_STRIDE>(M);
        static constexpr int OUTPUTS_A = N;
        static constexpr int OUTPUTS_B = O;

        struct Bucket
        {
            ALIGN int16_t output[OUTPUTS_A] = { };
            uint64_t hash = 0;
        };

        /* Per-thread cache: last accumulator computed in each bucket, plus its
         * piece bitboards, so bucket changes refresh by diffing pieces against
         * a same-bucket position instead of rebuilding all active inputs.
         */
        struct RefreshEntry
        {
            ALIGN int16_t output[OUTPUTS_A];
            Bitboard pieces[6][2] = { };
            bool turn = false;
            bool valid = false;
        };

        struct RefreshTable
        {
            RefreshEntry entry[NUM_BUCKETS];
        };

        /* Single slot trades the per-bucket output cache for a smaller footprint. */
        static constexpr int SLOTS = NNUE_SINGLE_BUCKET ? 1 : NUM_BUCKETS;

        Bucket _bucket[SLOTS];
        int _current_bucket = 0;
        ALIGN int16_t _output_b[OUTPUTS_B] = { };

        INLINE Bucket& slot(int bucket) { return _bucket[NNUE_SINGLE_BUCKET ? 0 : bucket]; }
        INLINE const Bucket& slot(int bucket) const { return _bucket[NNUE_SINGLE_BUCKET ? 0 : bucket]; }

    #if DEBUG_INCREMENTAL
        /* remember previous inputs, for debugging */
        ALIGN input_t _input[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)] = { }; /* one-hot encoding */

        /* full-width shadow of the per-bucket hashes, so the bucket-vs-hash
         * equivalence can be checked even when SLOTS == 1 (single-bucket mode)
         */
        uint64_t _ref_hash[NUM_BUCKETS] = { };
    #endif /* DEBUG_INCREMENTAL */


        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != slot(_current_bucket).hash;
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

            layer_1a.dot(_input, slot(bucket).output, base);
            layer_1b.dot(_input, _output_b);

            slot(bucket).hash = state.hash();
            _current_bucket = bucket;
        #if DEBUG_INCREMENTAL
            _ref_hash[bucket] = state.hash();
        #endif
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
        }

        /** Update 1st layer output incrementally, based on a previous state */
        template <typename LA, typename LB>
        INLINE void update(
            const LA& layer_a,
            const LB& layer_b,
            const State& prev,
            const State& state,
            const Move& move,
            Accumulator& ancestor,
            RefreshTable& refresh)
        {
            ASSERT(needs_update(state));
            ASSERT(ancestor.slot(ancestor._current_bucket).hash == prev.hash());

            const int bucket = get_bucket(state);
            const int prev_bucket = ancestor._current_bucket;
            bool incremental_a = (bucket == prev_bucket);

        #if DEBUG_INCREMENTAL
            /* bucket==prev_bucket must match the per-bucket-hash predicate; _ref_hash shadows it for SLOTS==1 */
            ASSERT_ALWAYS(incremental_a == (ancestor._ref_hash[bucket] == prev.hash()));
          #if !NNUE_SINGLE_BUCKET
            ASSERT_ALWAYS(ancestor._ref_hash[bucket] == ancestor._bucket[bucket].hash);
          #endif /* !NNUE_SINGLE_BUCKET */
        #endif /* DEBUG_INCREMENTAL */

        #if DEBUG_INCREMENTAL
            {
                static std::atomic<bool> seen[NUM_BUCKETS][NUM_BUCKETS];
                if (!seen[prev_bucket][bucket].exchange(true))
                    std::cerr << "[nnue] bucket " << prev_bucket << " -> " << bucket << (bucket == prev_bucket ? " same" : " cross") << std::endl;
            }
        #endif /* DEBUG_INCREMENTAL */

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

            /* Where the delta loops read the prior values from on the
             * same-bucket path; unused on the bucket-change path, which
             * refreshes slot(bucket).output from the refresh table.
             */
            const int16_t* src_a = ancestor.slot(bucket).output;

            if (incremental_a)
            {
                /* fused: incremental_update reads src_a, writes slot(bucket).output */
            }
            else if (slot(bucket).hash != state.hash())
            {
                auto& entry = refresh.entry[bucket];

                if (!entry.valid)
                {
                    /* empty-board entry: the diff below rebuilds from bias + all pieces */
                    memcpy(entry.output, layer_a._b, sizeof(entry.output));
                    entry.valid = true;
                }

                int rem[MAX_ACTIVE_INPUTS], add[MAX_ACTIVE_INPUTS];
                int nr = 0, na = 0;

                const Bitboard bbs[6] = { state.kings, state.pawns, state.knights, state.bishops, state.rooks, state.queens };
                int i = 63;
                for (int t = 0; t != 6; ++t)
                    for (int c = 0; c != 2; ++c)
                    {
                        const Bitboard bb = bbs[t] & state._occupied_co[c];
                        const Bitboard old = entry.pieces[t][c];
                        for_each_square_r(old & ~bb, [&](Square sq) { rem[nr++] = i - sq; });
                        for_each_square_r(bb & ~old, [&](Square sq) { add[na++] = i - sq; });
                        entry.pieces[t][c] = bb;
                        i += 64;
                    }

                if (state.turn != entry.turn)
                {
                    if (state.turn)
                        add[na++] = TURN_INDEX;
                    else
                        rem[nr++] = TURN_INDEX;
                    entry.turn = state.turn;
                }


            #if __ARM__
                using VecShort = Vec16s;
            #else
                using VecShort = Vec32s;
            #endif
                static_assert(OUTPUTS_A % VecShort::size() == 0);

                VecShort vo, vw;
                for (int j = 0; j != OUTPUTS_A; j += VecShort::size())
                {
                    vo.load_a(&entry.output[j]);
                    for (int k = 0; k != nr; ++k)
                    {
                        vw.load_a(&layer_a._w[base + rem[k]][j]);
                        vo -= vw;
                    }
                    for (int k = 0; k != na; ++k)
                    {
                        vw.load_a(&layer_a._w[base + add[k]][j]);
                        vo += vw;
                    }
                    vo.store_a(&entry.output[j]);
                    vo.store_a(&slot(bucket).output[j]);
                }
                /* layer A is final for state; incremental_update handles layer B only */
            }

            /* layer B: updated incrementally from the ancestor inside incremental_update */
            incremental_update(layer_a, layer_b, remove_inputs, add_inputs, r_idx, a_idx, base, bucket, incremental_a, src_a, ancestor._output_b);

            slot(bucket).hash = state.hash();
            _current_bucket = bucket;

        #if DEBUG_INCREMENTAL
            _ref_hash[bucket] = state.hash();

            // Validate that incremental_update produces same result as full dot products
            // layer A
            ALIGN int16_t output_a[OUTPUTS_A] = { };
            layer_a.dot(temp, output_a, base);
            for (int i = 0; i != OUTPUTS_A; ++i)
                ASSERT_ALWAYS(abs(output_a[i] - slot(bucket).output[i]) < 0.0001);

            // layer B
            ALIGN int16_t output_b[OUTPUTS_B] = { };
            layer_b.dot(temp, output_b);
            for (int i = 0; i != OUTPUTS_B; ++i)
                ASSERT_ALWAYS(abs(output_b[i] - _output_b[i]) < 0.0001);
        #endif /* DEBUG_INCREMENTAL */
        }

        /** Recompute incrementally.
         * src_a / src_b: where to read the prior accumulator values from. Reading
         * the ancestor directly in the delta loops (instead of memcpy-ing it into
         * this object first) saves a full write+read pass over the outputs.
         */
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
            bool update_layer_a,
            const int16_t* src_a,
            const int16_t* src_b)
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
                    vo.load_a(&src_a[j]);

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
                    vo.store_a(&slot(bucket).output[j]);
                }
            }

            if (update_layer_b)
            {
                /* Layer B */
                for (int j = 0; j != OUTPUTS_B; j += VecShort::size())
                {
                    vo.load_a(&src_b[j]);

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
            else if (src_b != _output_b)
            {
                memcpy(_output_b, src_b, sizeof(_output_b));
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


#if ATTACK_MASKS
    #if INSTRSET < 9
        using VTS = Vec8s;
        using VTI = Vec8i;
    #else
        using VTS = Vec16s;
        using VTI = Vec16i;
    #endif /* AVX512 */

    constexpr int THREAT_LANES = int(VTS::size());
    constexpr int THREAT_VECS = THREATS_OUT / THREAT_LANES;
    static_assert(THREATS_OUT % THREAT_LANES == 0);

    /* threat planes in H5 column order: (king,pawn,knight,bishop,rook,queen) x (black,white) */
    INLINE void threat_cols(const Bitboard (&planes)[2][7], Bitboard (&cols)[THREAT_PLANES])
    {
        int p = 0;
        for (auto t : { KING, PAWN, KNIGHT, BISHOP, ROOK, QUEEN })
            for (auto c : { BLACK, WHITE })
                cols[p++] = planes[c][t];
    }

    /* apply one plane's rows for the given bits: op(vsum[j], extended weight row chunk) */
    template <typename LC, typename OP>
    INLINE void threat_plane_rows(const LC& l1c, Bitboard bits, int plane, VTI (&vsum)[THREAT_VECS], OP op)
    {
        for_each_square(bits, [&](Square sq) {
            /* input-major: each _w row is a contiguous, 64-byte aligned int16[THREATS_OUT] */
            const auto& row = l1c._w[plane * 64 + 63 - sq];
            for (int j = 0; j != THREAT_VECS; ++j)
                op(vsum[j], extend(VTS().load_a(&row[j * THREAT_LANES])));
        });
    }

    /* full recompute of the hidden_1c pre-activation sums (bias + all active rows) */
    template <typename LC>
    INLINE void threat_refresh(const LC& l1c, const Bitboard (&planes)[2][7], int32_t (&sums)[THREATS_OUT])
    {
        Bitboard cols[THREAT_PLANES];
        threat_cols(planes, cols);

        VTI vsum[THREAT_VECS];
        for (int j = 0; j != THREAT_VECS; ++j)
            vsum[j] = extend(VTS().load_a(&l1c._b[j * THREAT_LANES]));

        for (int p = 0; p != THREAT_PLANES; ++p)
            threat_plane_rows(l1c, cols[p], p, vsum, [](VTI& a, VTI b) { a += b; });

        for (int j = 0; j != THREAT_VECS; ++j)
            vsum[j].store_a(&sums[j * THREAT_LANES]);
    }

    /* patch prev sums with only the changed plane bits (add new, subtract gone) */
    template <typename LC>
    INLINE void threat_update(
        const LC& l1c,
        const Bitboard (&prev_planes)[2][7],
        const Bitboard (&planes)[2][7],
        const int32_t (&prev_sums)[THREATS_OUT],
        int32_t (&sums)[THREATS_OUT])
    {
        Bitboard prev_cols[THREAT_PLANES], cols[THREAT_PLANES];
        threat_cols(prev_planes, prev_cols);
        threat_cols(planes, cols);

        VTI vsum[THREAT_VECS];
        for (int j = 0; j != THREAT_VECS; ++j)
            vsum[j].load_a(&prev_sums[j * THREAT_LANES]);

        for (int p = 0; p != THREAT_PLANES; ++p)
        {
            const auto changed = prev_cols[p] ^ cols[p];
            if (!changed)
                continue;
            threat_plane_rows(l1c, changed & cols[p], p, vsum, [](VTI& a, VTI b) { a += b; });
            threat_plane_rows(l1c, changed & prev_cols[p], p, vsum, [](VTI& a, VTI b) { a -= b; });
        }

        for (int j = 0; j != THREAT_VECS; ++j)
            vsum[j].store_a(&sums[j * THREAT_LANES]);

    #if DEBUG_INCREMENTAL
        int32_t check[THREATS_OUT];
        threat_refresh(l1c, planes, check);
        for (int i = 0; i != THREATS_OUT; ++i)
            ASSERT_ALWAYS(check[i] == sums[i]);
    #endif /* DEBUG_INCREMENTAL */
    }
#endif /* ATTACK_MASKS */


    template <typename A, typename P, typename L2, typename L3, typename OUT>
    INLINE int eval(const A& a, const P& pw, const L2& l2, const L3& l3, const OUT& out, bool turn
#if ATTACK_MASKS
        , const int32_t (&threat_sums)[THREATS_OUT]
#endif
    )
    {
        constexpr int POOL_OUT = A::OUTPUTS_A / POOL_STRIDE;
        static_assert(P::param_count() == 2 * A::OUTPUTS_A);
#if ATTACK_MASKS
        static_assert(POOL_OUT + THREATS_OUT == L2::INPUTS); /* hidden_1c concats into L2 */
#else
        static_assert(POOL_OUT == L2::INPUTS);
#endif
        static_assert(A::OUTPUTS_B == POOL_OUT); /* 1b modulates pooled 1:1 */
        static_assert(L2::OUTPUTS == L3::INPUTS);
        static_assert(L3::OUTPUTS == OUT::INPUTS);

        ALIGN float l2_in[L2::INPUTS];
        ALIGN float l2_out[L2::OUTPUTS];
        ALIGN float l3_out[L3::OUTPUTS];
        ALIGN float output[1]; // eval

        /* pooled + modulation fill the first POOL_OUT entries of l2_in */
        pool(a.slot(a._current_bucket).output, pw._w[turn], reinterpret_cast<float(&)[POOL_OUT]>(l2_in));

        static_assert(POOL_OUT % Vector::size() == 0);

        /* hidden_1b (linear) modulates pooled: l2_in *= (1 + dequant(_output_b)) */
        const Vector v_one(1.0f);

#if INSTRSET >= 7
        /* AVX/AVX2/AVX512: short-vector width matches Vector; fuse load+widen+modulate. */
    #if INSTRSET >= 9
        using VS = Vec16s;
    #else
        using VS = Vec8s;
    #endif
        static_assert(VS::size() == Vector::size());

        const Vector v_scale(1.0f / QSCALE);
        for (int i = 0; i != POOL_OUT; i += Vector::size())
        {
            Vector v1;
            v1.load_a(&l2_in[i]);
            const Vector m = to_float(extend(VS().load_a(&a._output_b[i]))) * v_scale + v_one;
            (v1 * m).store_a(&l2_in[i]);
        }
#else
        /* SSE2: short width (8) != Vector width (4); dequantize to scratch first. */
        ALIGN float mod[POOL_OUT];
        dequantize(a._output_b, mod);

        for (int i = 0; i != POOL_OUT; i += Vector::size())
        {
            Vector v1, m;
            v1.load_a(&l2_in[i]);
            m.load_a(&mod[i]);
            (v1 * (m + v_one)).store_a(&l2_in[i]);
        }
#endif /* INSTRSET >= 7 */

#if ATTACK_MASKS
        /* hidden_1c: activate the incrementally maintained sums into the L2 tail */
        activate(threat_sums, reinterpret_cast<float(&)[THREATS_OUT]>(l2_in[POOL_OUT]));
#endif /* ATTACK_MASKS */

        l2.dot(l2_in, l2_out, [](const Vector& v) { return relu(v); });
        l3.dot(l2_out, l3_out, [](const Vector& v) { return relu(v); });
        out.dot(l3_out, output);
        return EVAL_SCALE * output[0];
    }


    /* Compute the move head's sub-accumulator once per node: relu(bias + W_acc . active),
     * kept in the quantized int16 domain (like the eval accumulator, no dequantize). Sparse
     * gather over active inputs; the result feeds score_move for every move at this node.
     * LMA is the [ACTIVE_INPUTS x MOVE_ACC] quantized layer.
     */
    template <typename LMA, size_t N>
    INLINE void move_accumulate(
        const LMA& layer_acc, const int (&active)[MAX_ACTIVE_INPUTS], int count, int16_t (&acc)[N])
    {
        for (size_t j = 0; j < N; ++j)
        {
            int sum = layer_acc._b[j];  /* int16 weights/bias, accumulate in int32 */
            for (int k = 0; k < count; ++k)
                sum += layer_acc._wt[j][active[k]];
            /* relu; keep at QSCALE scale (like the eval accumulator), clamp to int16 */
            sum = std::min<int>(std::numeric_limits<int16_t>::max(), std::max(0, sum));
            acc[j] = int16_t(sum);
        }
    }

    /* Per-move logit: bias[index] + acc . W_move[:, index]. LM is the [MOVE_ACC x 4096]
     * quantized layer, so _wt[index] is the length-MOVE_ACC column for this (from,to) move.
     * Stays integer: the 256 int16xint16 products need int64 (worst case ~2.7e11), then a
     * fixed shift back into int16 range. Move scores are compared only against each other
     * (Phase 4 LATE_MOVES), so the constant scale is irrelevant; only relative order matters.
     */
    template <typename LM, size_t N>
    INLINE void score_move(const LM& layer_m, const int16_t (&acc)[N], Move& move)
    {
        const auto index = move.from_square() * 64 + move.to_square();

        int64_t score = int64_t(layer_m._b[index]) << QLOG2;  /* match the acc.wt product scale */
        for (size_t j = 0; j < N; ++j)
            score += int64_t(acc[j]) * int64_t(layer_m._wt[index][j]);

        score >>= QLOG2;  /* one QSCALE factor out; keep ordering, fit int16 */
        using move_score_t = decltype(move._score);
        score = std::min<int64_t>(std::numeric_limits<move_score_t>::max(), score);
        move._score = move_score_t(std::max<int64_t>(std::numeric_limits<move_score_t>::lowest(), score));
    }
} /* namespace nnue */
