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
#include <algorithm>
#include <istream>
#include <stdexcept>
#include <string>
#include <type_traits>

#if (__amd64__) || (__x86_64__) || (__i386__) || (_M_AMD64) || (_M_X64) || (_M_IX86)
    #include "x86vector.h"
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

namespace nnue
{
    static const std::string instrset = ARCH ARCH_FMA ARCH_VNNI ARCH_BF16;

    using namespace chess;
    using input_t = int16_t;
    using weight_t = float;

    constexpr int ACTIVE_INPUTS = 768; /* piece-square, white view */
    constexpr int EVAL_SCALE = 100;
    constexpr int MAX_ACTIVE_INPUTS = 32; /* one per piece */
    constexpr int NUM_BUCKETS = 32; /* king bucket x opponent queen */
    constexpr int QSCALE = 1024;

    /* own king square, a1 = 0 -> king bucket */
    constexpr int KING_BUCKETS[64] = {
         0,  1,  2,  3,  3,  2,  1,  0,
         4,  5,  6,  7,  7,  6,  5,  4,
         8,  9, 10, 11, 11, 10,  9,  8,
         8,  9, 10, 11, 11, 10,  9,  8,
        12, 12, 13, 13, 13, 13, 12, 12,
        12, 12, 13, 13, 13, 13, 12, 12,
        14, 14, 15, 15, 15, 15, 14, 14,
        14, 14, 15, 15, 15, 15, 14, 14,
    };

    static_assert(std::all_of(std::begin(KING_BUCKETS), std::end(KING_BUCKETS), [](int b) { return b >= 0 && b < NUM_BUCKETS / 2; }));

    /* black-view input index: color swap (64) + rank flip (56) */
    constexpr int PERSPECTIVE_XOR = 120;

    /* mirrored-view input index: file flip */
    constexpr int MIRROR_XOR = 7;

    /* per view: L1A block x 2 + mirrored */
    constexpr int VIEW_KEYS = NUM_BUCKETS * 2;

    /* accumulator rows: 32 pieces + bias must not overflow int16 */
    constexpr int ACC_WEIGHT_MAX = INT16_MAX / (MAX_ACTIVE_INPUTS + 1);

    /* Activation: multiply pairs of accumulator values capped to [0, 1023], giving bytes 0..127 (1.0 == 128) */

    constexpr int ACT_SHIFT = 3;
    constexpr int ACT_MAX = 127;
    constexpr int ACT_CLAMP = QSCALE - 1;
    constexpr int ACT_SCALE = QSCALE >> ACT_SHIFT;

    /* own king on files e-h mirrors the view */
    INLINE int view_key(const State& state, Color view)
    {
        const int sq = view == WHITE ? state.king(WHITE) : square_mirror(state.king(BLACK));
        const int mirrored = square_file(sq) >= 4;
        const int queen = bool(state.queens & state.occupied_co(!view));
        return (KING_BUCKETS[mirrored ? sq ^ MIRROR_XOR : sq] * 2 + queen) * 2 + mirrored;
    }

    INLINE int piece_bucket(const State& state)
    {
        return (popcount(state.occupied()) - 1) / 4;
    }

    INLINE int stack_index(const State& state)
    {
        return piece_bucket(state) * 2 + state.turn;
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
    }

    template <typename F>
    static INLINE void for_each_piece_input(const State& state, F&& func)
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
    }

    /** Calculate the piece-square index into the one-hot encoding. */
    INLINE constexpr int piece_square_index(PieceType piece_type, Color color, Square square)
    {
        return (piece_type % 6) * 128 + (64 * color) + 63 - square;
    }


    /** Rectified Linear Unit (reLU) activation */
    INLINE Vector relu(Vector v) { return max(v, v_zero); }


    /* SIMD types for the int8 layer: accumulator (int16), activations (u8), weights (s8), sums (int32) */
#if INSTRSET >= 10 /* AVX-512BW */
    using VecS16 = Vec32s;
    using VecU8 = Vec64uc;
    using VecS8 = Vec64c;
    using VecI32 = Vec16i;
#elif INSTRSET >= 8
    using VecS16 = Vec16s;
    using VecU8 = Vec32uc;
    using VecS8 = Vec32c;
    using VecI32 = Vec8i;
#else
    using VecS16 = Vec8s;
    using VecU8 = Vec16uc;
    using VecS8 = Vec16c;
    using VecI32 = Vec4i;
#endif /* INSTRSET */

    /* independent sets of sums in the sparse dot, so consecutive dot_adds don't wait on each other */
    constexpr int DOT_SETS = (INSTRSET >= 10) ? 4 : (INSTRSET >= 8) ? 2 : 1;


#if 0 /* superseded by the non-zero blocks recorded in activate */
    /** Indices of the 4-byte input blocks that are not all zero, in order; returns their count */
    template <int N>
    INLINE int find_nonzero_blocks(const uint8_t (&in)[N], int32_t (&blocks)[N / 4])
    {
        constexpr int W = VecI32::size();
        static_assert(N % (4 * W) == 0);

        int count = 0;
        for (int i = 0; i != N / 4; i += W)
            count = append_nonzero_lanes(blocks, count, i, VecI32().load_a(&in[4 * i]));
        return count;
    }
#endif /* superseded */


    /* Per view: multiply the capped halves pairwise into bytes 0..127; lists the non-zero 4-byte blocks, returns their count */
    template <int N>
    INLINE int activate(const int16_t (&in)[2][N], uint8_t (&out)[N], int32_t (&blocks)[N / 4])
    {
        constexpr int W = VecS16::size();
        constexpr int HALF = N / 2;
        static_assert(2 * W == VecU8::size() && HALF % (2 * W) == 0 && VecI32::size() * 4 == VecU8::size());

        const VecS16 zero(0), top(ACT_CLAMP);

        /* (a << ACT_SHIFT) * b >> 16 must rescale 1/QSCALE^2 to 1/ACT_SCALE */
        static_assert((1 << (16 - ACT_SHIFT)) == QSCALE << ACT_SHIFT);

        /* b is not capped below: with a >= 0, a negative b gives a product <= 0, which the unsigned pack turns into 0 */
        const auto product = [&](const int16_t (&half)[N], int i)
        {
            const VecS16 a = min(max(VecS16().load_a(&half[i]), zero), top) << ACT_SHIFT;
            const VecS16 b = min(VecS16().load_a(&half[i + HALF]), top);
            return mul_hi(a, b);
        };

        int count = 0;
        for (int view = 0; view != 2; ++view)
            for (int i = 0; i != HALF; i += 2 * W)
            {
                const VecU8 bytes = compress_saturated_s2u(product(in[view], i), product(in[view], i + W));
                bytes.store_a(&out[view * HALF + i]);
                count = append_nonzero_lanes(blocks, count, (view * HALF + i) / 4, VecI32(bytes));
            }
        return count;
    }


    /* int8 weights accumulate in int32, so their biases are int32 */
    template <typename T> using bias_type = std::conditional_t<std::is_same_v<T, int8_t>, int32_t, T>;

    template <int I, int O, typename T, int INPUTS, bool Incremental>
    struct BaseLayer
    {
        ALIGN bias_type<T> _b[O]; /* biases */
        ALIGN T _wt[O][INPUTS]; /* weights transposed */
    };


    template <int I, int O, int INPUTS>
    struct BaseLayer<I, O, int8_t, INPUTS, false>
    {
        static_assert(INPUTS % 4 == 0);

        ALIGN int32_t _b[O]; /* biases */
        ALIGN int8_t _packed[INPUTS / 4][O][4]; /* weights in the order dot() reads them */
    };


    template <int I, int O, typename T, int INPUTS>
    struct BaseLayer<I, O, T, INPUTS, true>
    {
        ALIGN bias_type<T> _b[O]; /* biases */
        ALIGN T _w[I][O]; /* one row per input, for incremental updates */
    };


    /* Weights are float on disk, quantized at load: weights at Scale, biases at Scale x InScale (the input scale) */
    template <int I, int O, typename T=weight_t, int Scale=1, bool Incremental=false, int InScale=1>
    struct Layer : BaseLayer<I, O, T, (Scale == 1 || Incremental) ? I : round_up<INPUT_STRIDE>(I), Incremental>
    {
        using bias_t = bias_type<T>;

        /* Round up to INPUT_STRIDE to deal with odd inputs. */
        static constexpr int INPUTS = (Scale == 1 || Incremental) ? I : round_up<INPUT_STRIDE>(I);
        static constexpr int OUTPUTS = O;
        static constexpr int BIAS_SCALE = Scale * InScale;

        /* int8 weights by blocks of 4 inputs, so dot() can skip blocks whose activations are all zero */
        static constexpr bool PACKED = std::is_same_v<T, int8_t> && !Incremental;

        INLINE T& packed(int j, int i)
        {
            return this->_packed[i / 4][j][i % 4];
        }

        Layer() = default;

        Layer(const float(&w)[I][OUTPUTS], const float(&b)[OUTPUTS])
        {
            set_weights(w, b);
        }

        static constexpr size_t param_count()
        {
            return (I + 1) * O;
        }

        template <typename V>
        static V quantize(float v, int scale)
        {
            if constexpr (Scale == 1)
                return v;
            else
            {
                const auto q = std::round(double(v) * scale);
                /* symmetric: int8 -128 is rejected */
                if (std::abs(q) > std::numeric_limits<V>::max())
                    throw std::runtime_error("weight " + std::to_string(v) + " exceeds range at scale " + std::to_string(scale));
                return V(q);
            }
        }

        void set_weights(const float(&w)[I][OUTPUTS], const float(&b)[OUTPUTS])
        {
            for (int j = 0; j != OUTPUTS; ++j)
                this->_b[j] = quantize<bias_t>(b[j], BIAS_SCALE);

            for (int i = 0; i != I; ++i)
            {
                for (int j = 0; j != OUTPUTS; ++j)
                {
                    const T v = quantize<T>(w[i][j], Scale);
                    if constexpr (Incremental)
                        this->_w[i][j] = v;
                    else if constexpr (PACKED)
                        packed(j, i) = v;
                    else
                        this->_wt[j][i] = v;
                }
            }
            /* padding, if needed */
            if constexpr (!Incremental)
                for (int i = I; i != INPUTS; ++i)
                    for (int j = 0; j != OUTPUTS; ++j)
                    {
                        if constexpr (PACKED)
                            packed(j, i) = 0;
                        else
                            this->_wt[j][i] = 0;
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

        /** Byte activations times int8 weights over the non-zero input blocks, summed in int32, then relu and back to float */
        INLINE void dot(const uint8_t (&input)[INPUTS], const int32_t (&blocks)[INPUTS / 4], int count, float (&output)[OUTPUTS]) const
        {
            static_assert(PACKED);

            constexpr int W = VecI32::size();
            constexpr int ACCS = OUTPUTS / W;
            static_assert(OUTPUTS % W == 0 && VecS8::size() == 4 * W);
            constexpr float OUT_SCALE = 1.0f / BIAS_SCALE;

            VecI32 acc[DOT_SETS][ACCS];
            for (int s = 0; s != DOT_SETS; ++s)
                for (int k = 0; k != ACCS; ++k)
                    acc[s][k] = VecI32(0);

            const auto accumulate = [&](VecI32 (&sums)[ACCS], int b)
            {
                int32_t four;
                memcpy(&four, &input[4 * b], 4);
                const VecU8 a = VecU8(VecI32(four));

                VecS8 w;
                for (int k = 0; k != ACCS; ++k)
                {
                    w.load_a(this->_packed[b][k * W]);
                    sums[k] = dot_add(sums[k], a, w);
                }
            };

            int n = 0;
            for (; n + DOT_SETS <= count; n += DOT_SETS)
                for (int s = 0; s != DOT_SETS; ++s)
                    accumulate(acc[s], blocks[n + s]);
            for (; n != count; ++n)
                accumulate(acc[0], blocks[n]);

            ALIGN int32_t sums[OUTPUTS];
            for (int k = 0; k != ACCS; ++k)
            {
                for (int s = 1; s != DOT_SETS; ++s)
                    acc[0][k] += acc[s][k];
                acc[0][k].store_a(&sums[k * W]);
            }

            for (int j = 0; j != OUTPUTS; ++j)
                output[j] = float(std::max(0, sums[j] + this->_b[j])) * OUT_SCALE;
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
            dot(input, output, this->_b, this->_wt, [](const Vector& v) { return v; }, 0);
        }

        template <size_t N, typename U, typename V, typename ACTIVATION>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS], ACTIVATION activate) const
        {
            dot(input, output, this->_b, this->_wt, activate, 0);
        }
    };


    template <int M, int N> struct Accumulator
    {
        static_assert(ACTIVE_INPUTS * NUM_BUCKETS == M);

        static constexpr int HALF = N; /* one perspective */
        static constexpr int OUTPUTS = 2 * N; /* [black view, white view] */

    #if __ARM__
        using VecShort = Vec16s;
    #else
        using VecShort = Vec32s;
    #endif /* __ARM__ */
        static_assert(HALF % VecShort::size() == 0);

        /* Per-thread cache: last half computed for each view and key, plus its
         * piece bitboards, so key changes refresh by diffing pieces against
         * a same-key position instead of rebuilding all active inputs.
         */
        struct RefreshEntry
        {
            ALIGN int16_t output[HALF];
            Bitboard pieces[6][2] = { };
            bool valid = false;
        };

        struct RefreshTable
        {
            RefreshEntry entry[2][VIEW_KEYS];
        };

        ALIGN int16_t _output[2][HALF] = { }; /* [black view, white view] */
        uint64_t _hash = 0;
        int _key[2] = { };

        INLINE const int16_t (&output() const)[2][HALF] { return _output; }

    #if DEBUG_INCREMENTAL
        /* remember previous inputs, for debugging */
        ALIGN input_t _input[ACTIVE_INPUTS] = { }; /* one-hot encoding */
    #endif /* DEBUG_INCREMENTAL */


        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != _hash;
        }

        /* 32 pieces + bias per output must not overflow int16 */
        template <typename LA>
        static void check_weights(const LA& layer)
        {
            const auto check = [](int v)
            {
                if (std::abs(v) > ACC_WEIGHT_MAX)
                    throw std::runtime_error("accumulator weight " + std::to_string(v) + " exceeds " + std::to_string(ACC_WEIGHT_MAX));
            };
            for (const auto b : layer._b)
                check(b);
            for (const auto& row : layer._w)
                for (const auto w : row)
                    check(w);
        }


        /* piece count known only at run time */
        static constexpr int DYNAMIC = -1;

        /* empty delta list */
        static constexpr int NONE[MAX_ACTIVE_INPUTS] = {};

        static INLINE int view_xor(Color view, int key)
        {
            return (key & 1 ? MIRROR_XOR : 0) ^ (view == WHITE ? 0 : PERSPECTIVE_XOR);
        }

        /* Update one perspective: subtract the rows of removed pieces, add the new ones; with Copy, out2 gets a copy too.
         * REMOVED and ADDED fix the piece counts at compile time, which turns the loops into straight-line code.
         *
         *  piece i (white-view index), view key = block * 2 + mirrored:
         *    white half  +=  layer._w[ block * 768 + (i ^ m)       ]   (HALF values)
         *    black half  +=  layer._w[ block * 768 + (i ^ 120 ^ m) ]   (HALF values)
         *  m: file flip (7) if mirrored, else 0; 120: color swap (64) + rank flip (56)
         */
        template <int REMOVED = DYNAMIC, int ADDED = DYNAMIC, bool Copy = false, typename LA>
        static INLINE void apply_deltas(
            const LA& layer,
            Color view,
            int key,
            const int16_t (&src)[HALF],
            int16_t (&out)[HALF],
            int16_t (&out2)[HALF],
            const int (&remove)[MAX_ACTIVE_INPUTS],
            int r_idx,
            const int (&add)[MAX_ACTIVE_INPUTS],
            int a_idx)
        {
            static_assert(LA::OUTPUTS == HALF);

            const int removed = (REMOVED == DYNAMIC) ? r_idx : REMOVED;
            const int added = (ADDED == DYNAMIC) ? a_idx : ADDED;
            ASSERT(removed == r_idx && added == a_idx);

            const size_t base = size_t(key >> 1) * ACTIVE_INPUTS;
            const int x = view_xor(view, key);

            VecShort acc, v;
            for (int j = 0; j != HALF; j += VecShort::size())
            {
                acc.load_a(&src[j]);

                for (int i = 0; i < removed; ++i)
                {
                    v.load_a(&layer._w[base + (remove[i] ^ x)][j]);
                    acc -= v;
                }

                for (int i = 0; i < added; ++i)
                {
                    v.load_a(&layer._w[base + (add[i] ^ x)][j]);
                    acc += v;
                }

                acc.store_a(&out[j]);
                if constexpr (Copy)
                    acc.store_a(&out2[j]);
            }
        }


        /* Both perspectives in one pass, for moves that change neither view key; not shared with the per-view version, which measured slower */
        template <int REMOVED = DYNAMIC, int ADDED = DYNAMIC, typename LA>
        static INLINE void apply_deltas(
            const LA& layer,
            const int (&key)[2],
            const int16_t (&src)[2][HALF],
            int16_t (&out)[2][HALF],
            const int (&remove)[MAX_ACTIVE_INPUTS],
            int r_idx,
            const int (&add)[MAX_ACTIVE_INPUTS],
            int a_idx)
        {
            static_assert(LA::OUTPUTS == HALF);

            const int removed = (REMOVED == DYNAMIC) ? r_idx : REMOVED;
            const int added = (ADDED == DYNAMIC) ? a_idx : ADDED;
            ASSERT(removed == r_idx && added == a_idx);

            const size_t base_b = size_t(key[BLACK] >> 1) * ACTIVE_INPUTS;
            const size_t base_w = size_t(key[WHITE] >> 1) * ACTIVE_INPUTS;
            const int x_b = view_xor(BLACK, key[BLACK]);
            const int x_w = view_xor(WHITE, key[WHITE]);

            VecShort vb, vw, v;
            for (int j = 0; j != HALF; j += VecShort::size())
            {
                vb.load_a(&src[BLACK][j]);
                vw.load_a(&src[WHITE][j]);

                for (int i = 0; i < removed; ++i)
                {
                    v.load_a(&layer._w[base_b + (remove[i] ^ x_b)][j]);
                    vb -= v;
                    v.load_a(&layer._w[base_w + (remove[i] ^ x_w)][j]);
                    vw -= v;
                }

                for (int i = 0; i < added; ++i)
                {
                    v.load_a(&layer._w[base_b + (add[i] ^ x_b)][j]);
                    vb += v;
                    v.load_a(&layer._w[base_w + (add[i] ^ x_w)][j]);
                    vw += v;
                }

                vb.store_a(&out[BLACK][j]);
                vw.store_a(&out[WHITE][j]);
            }
        }


        /* Update after a move; the usual moves get their own compile-time version of apply_deltas */
        template <typename LA>
        static INLINE void apply_move(
            const LA& layer,
            Color view,
            int key,
            const int16_t (&src)[HALF],
            int16_t (&out)[HALF],
            const int (&remove)[MAX_ACTIVE_INPUTS],
            int removed,
            const int (&add)[MAX_ACTIVE_INPUTS],
            int added)
        {
            /* out2 is unused without Copy */
            if (removed == 1 && added == 1) /* quiet move or promotion */
                apply_deltas<1, 1>(layer, view, key, src, out, out, remove, removed, add, added);
            else if (removed == 2 && added == 1) /* capture */
                apply_deltas<2, 1>(layer, view, key, src, out, out, remove, removed, add, added);
            else if (removed == 2 && added == 2) /* castling */
                apply_deltas<2, 2>(layer, view, key, src, out, out, remove, removed, add, added);
            else
                apply_deltas(layer, view, key, src, out, out, remove, removed, add, added);
        }

        template <typename LA>
        static INLINE void apply_move(
            const LA& layer,
            const int (&key)[2],
            const int16_t (&src)[2][HALF],
            int16_t (&out)[2][HALF],
            const int (&remove)[MAX_ACTIVE_INPUTS],
            int removed,
            const int (&add)[MAX_ACTIVE_INPUTS],
            int added)
        {
            if (removed == 1 && added == 1) /* quiet move or promotion */
                apply_deltas<1, 1>(layer, key, src, out, remove, removed, add, added);
            else if (removed == 2 && added == 1) /* capture */
                apply_deltas<2, 1>(layer, key, src, out, remove, removed, add, added);
            else if (removed == 2 && added == 2) /* castling */
                apply_deltas<2, 2>(layer, key, src, out, remove, removed, add, added);
            else
                apply_deltas(layer, key, src, out, remove, removed, add, added);
        }


        /** Compute 1st layer output from scratch at root */
        template <typename LA>
        INLINE void full_update(const LA& layer, const State& state)
        {
            int add_inputs[MAX_ACTIVE_INPUTS];
            int a_idx = 0;
            for_each_piece_input(state, [&](int i) { add_inputs[a_idx++] = i; });

            for (const auto view : { BLACK, WHITE })
            {
                _key[view] = view_key(state, view);
                apply_deltas(layer, view, _key[view], layer._b, _output[view], _output[view], NONE, 0, add_inputs, a_idx);
            }

            _hash = state.hash();
        #if DEBUG_INCREMENTAL
            memset(&_input, 0, sizeof(_input));
            one_hot_encode(state, _input);
        #endif
        }

        template <typename LA>
        INLINE void update(const LA& layer, const State& state)
        {
            if (needs_update(state))
            {
                full_update(layer, state);
            }
        }

        /** Rebuild one half by diffing pieces against the last position seen with the same view key */
        template <typename LA>
        INLINE void refresh_view(const LA& layer, const State& state, Color view, int key, RefreshEntry& entry)
        {
            if (!entry.valid)
            {
                /* empty-board entry: the diff below rebuilds from bias + all pieces */
                memcpy(entry.output, layer._b, sizeof(layer._b));
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

            apply_deltas<DYNAMIC, DYNAMIC, true>(layer, view, key, entry.output, entry.output, _output[view], rem, nr, add, na);
        }

        /** Utility for incremental updates */
        static INLINE void delta(int (&d)[MAX_ACTIVE_INPUTS], int& idx, PieceType pt, Color col, Square sq)
        {
            d[idx++] = piece_square_index(pt, col, sq);
        }

        /** Update 1st layer output incrementally, based on a previous state */
        template <typename LA>
        INLINE void update(
            const LA& layer,
            const State& prev,
            const State& state,
            const Move& move,
            Accumulator& ancestor,
            RefreshTable& refresh)
        {
            ASSERT(needs_update(state));
            ASSERT(ancestor._hash == prev.hash());
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

            ALIGN input_t temp[ACTIVE_INPUTS] = { };
            one_hot_encode(state, temp);

            for (int i = 0; i != ACTIVE_INPUTS; ++i)
                ASSERT_ALWAYS(_input[i] == temp[i]);
        #endif /* DEBUG_INCREMENTAL */

            int key[2] = { ancestor._key[BLACK], ancestor._key[WHITE] };
            if (move)
            {
                if ((prev.kings & BB_SQUARES[move.from_square()]) || (prev.queens & BB_SQUARES[move.to_square()]))
                    key[prev.turn] = view_key(state, prev.turn);
                if (move.promotion() == PieceType::QUEEN)
                    key[!prev.turn] = view_key(state, !prev.turn);
            }

        #if DEBUG_INCREMENTAL
            for (const auto view : { BLACK, WHITE })
            {
                const int prev_key = ancestor._key[view];
                static std::atomic<bool> seen[2][VIEW_KEYS][VIEW_KEYS];
                if (!seen[view][prev_key][key[view]].exchange(true))
                    std::cerr << "[nnue] view " << int(view) << " key " << prev_key << " -> " << key[view] << (key[view] == prev_key ? " same" : " cross") << std::endl;
            }
        #endif /* DEBUG_INCREMENTAL */

            if (key[BLACK] == ancestor._key[BLACK] && key[WHITE] == ancestor._key[WHITE])
            {
                apply_move(layer, key, ancestor._output, _output, remove_inputs, r_idx, add_inputs, a_idx);
            }
            else
            {
                for (const auto view : { BLACK, WHITE })
                {
                    if (key[view] == ancestor._key[view])
                        apply_move(layer, view, key[view], ancestor._output[view], _output[view], remove_inputs, r_idx, add_inputs, a_idx);
                    else
                        refresh_view(layer, state, view, key[view], refresh.entry[view][key[view]]);
                }
            }

            _key[BLACK] = key[BLACK];
            _key[WHITE] = key[WHITE];

            _hash = state.hash();

        #if DEBUG_INCREMENTAL
            // Validate the incremental result against a full recompute
            Accumulator full;
            full.full_update(layer, state);
            for (const auto view : { BLACK, WHITE })
            {
                ASSERT_ALWAYS(full._key[view] == _key[view]);
                for (int i = 0; i != HALF; ++i)
                    ASSERT_ALWAYS(full._output[view][i] == _output[view][i]);
            }
        #endif /* DEBUG_INCREMENTAL */
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


    template <typename A, typename L2, typename L3, typename OUT>
    INLINE int eval(const A& a, const L2& l2, const L3& l3, const OUT& out)
    {
        static_assert(A::OUTPUTS / 2 == L2::INPUTS);
        static_assert(L2::OUTPUTS == L3::INPUTS);
        static_assert(L3::OUTPUTS == OUT::INPUTS);

        ALIGN uint8_t l2_in[A::OUTPUTS / 2];
        ALIGN float l2_out[L2::OUTPUTS];
        ALIGN float l3_out[L3::OUTPUTS];
        ALIGN float output[1]; // eval

        int32_t blocks[L2::INPUTS / 4];
        const int count = activate(a.output(), l2_in, blocks);
        l2.dot(l2_in, blocks, count, l2_out);
        l3.dot(l2_out, l3_out, [](const Vector& v) { return relu(v); });
        out.dot(l3_out, output);
        return EVAL_SCALE * output[0];
    }
} /* namespace nnue */
