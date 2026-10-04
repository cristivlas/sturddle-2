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

#ifndef NNUE_SINGLE_BUCKET
    #define NNUE_SINGLE_BUCKET true
#endif

namespace nnue
{
    static const std::string instrset = ARCH ARCH_FMA ARCH_VNNI ARCH_BF16;

    using namespace chess;
    using input_t = int16_t;
    using weight_t = float;

    constexpr int ACTIVE_INPUTS = 768; /* piece-square, white view */
    constexpr int EVAL_SCALE = 100;
    constexpr int MAX_ACTIVE_INPUTS = 33; // 32 pieces + turn (move head)
    constexpr int NUM_BUCKETS = 16;
    constexpr int PAWN_BUCKETS = chess::PAWN_BUCKETS;
    constexpr int KING_BUCKETS = 4;
    static_assert(NUM_BUCKETS == PAWN_BUCKETS * KING_BUCKETS, "bucket grid mismatch");
    constexpr int QSCALE = 1024;
    constexpr int QLOG2 = 10;  /* log2(QSCALE), for shift-based requantization */
    static_assert((1 << QLOG2) == QSCALE, "QLOG2 must be log2(QSCALE)");

    /* black-view input index: color swap (64) + rank flip (56) */
    constexpr int PERSPECTIVE_XOR = 120;

    /* accumulator rows: 32 pieces + bias must not overflow int16 */
    constexpr int ACC_WEIGHT_MAX = INT16_MAX / 33;

    /* activation: clamp(acc, 0, ACT_CLAMP) >> ACT_SHIFT -> u8 in [0, ACT_MAX] */
    constexpr int ACT_SHIFT = 3;
    constexpr int ACT_MAX = 127;
    constexpr int ACT_CLAMP = ((ACT_MAX + 1) << ACT_SHIFT) - 1;

    /* hidden_2: s8 weights; int32 sums and bias at activation scale x weight scale */
    constexpr int L2_WSCALE = 64;
    constexpr int L2_SCALE = (QSCALE >> ACT_SHIFT) * L2_WSCALE;

    /* move head inputs: piece-square + side to move */
    constexpr int MOVE_INPUTS = 769;
    constexpr int TURN_INDEX = 768;

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

    /* black-view bucket: the color swap exchanges the two king bits */
    INLINE constexpr int mirror_bucket(int bucket)
    {
        return (bucket & ~3) | ((bucket & 1) << 1) | ((bucket >> 1) & 1);
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

    /* move head inputs */
    template <typename F>
    static INLINE void for_each_active_input(const State& state, F&& func)
    {
        for_each_piece_input(state, func);

        if (state.turn)
            func(TURN_INDEX);
    }

    /** Calculate the piece-square index into the one-hot encoding. */
    INLINE constexpr int piece_square_index(PieceType piece_type, Color color, Square square)
    {
        return (piece_type % 6) * 128 + (64 * color) + 63 - square;
    }


    /** Rectified Linear Unit (reLU) activation */
    INLINE Vector relu(Vector v) { return max(v, v_zero); }


    template <int I, int O, typename T=weight_t, int Scale=1>
    struct Layer
    {
        static constexpr int ROWS = I;
        static constexpr int COLS = O;
        static constexpr int INPUTS = (Scale == 1) ? I : round_up<INPUT_STRIDE>(I);
        static constexpr int OUTPUTS = O;
        static constexpr int SCALE = Scale;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */

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
                    {
                        const auto q = std::round(w[i][j] * Scale);
                        if (q > std::numeric_limits<T>::max() || q < std::numeric_limits<T>::lowest())
                            throw std::runtime_error("weight " + std::to_string(w[i][j]) + " exceeds range at scale " + std::to_string(Scale));
                        v = q;
                    }
                    _wt[j][i] = v;
                }
            }
            /* padding, if needed */
            for (int i = I; i != INPUTS; ++i)
                for (int j = 0; j != OUTPUTS; ++j)
                    _wt[j][i] = 0;
        }

        void load_weights(std::istream& file)
        {
            auto w = std::make_unique<float[]>(I * OUTPUTS);
            auto b = std::make_unique<float[]>(OUTPUTS);

            file.read(reinterpret_cast<char*>(w.get()), I * OUTPUTS * sizeof(float));
            file.read(reinterpret_cast<char*>(b.get()), OUTPUTS * sizeof(float));

            set_weights(reinterpret_cast<float(&)[I][OUTPUTS]>(*(w.get())), reinterpret_cast<float(&)[OUTPUTS]>(*(b.get())));
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
            dot(input, output, _b, _wt, [](const Vector& v) { return v; }, 0);
        }

        template <size_t N, typename U, typename V, typename ACTIVATION>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS], ACTIVATION activate) const
        {
            dot(input, output, _b, _wt, activate, 0);
        }
    };


    /* Accumulator weights: one int16 row per (bucket, input) at QSCALE, shared by both perspectives */
    template <int I, int O>
    struct AccumulatorLayer
    {
        static constexpr int ROWS = I;
        static constexpr int OUTPUTS = O;

        ALIGN int16_t _b[OUTPUTS];
        ALIGN int16_t _w[ROWS][OUTPUTS];

        static constexpr size_t param_count()
        {
            return (I + 1) * O;
        }

        void load_weights(std::istream& file)
        {
            auto w = std::make_unique<float[]>(size_t(I) * O);
            auto b = std::make_unique<float[]>(O);

            file.read(reinterpret_cast<char*>(w.get()), size_t(I) * O * sizeof(float));
            file.read(reinterpret_cast<char*>(b.get()), O * sizeof(float));

            const auto quantize = [](float v)
            {
                const auto q = std::round(v * QSCALE);
                if (std::abs(q) > ACC_WEIGHT_MAX)
                    throw std::runtime_error("accumulator weight " + std::to_string(v) + " exceeds " + std::to_string(ACC_WEIGHT_MAX) + " at scale " + std::to_string(QSCALE));
                return int16_t(q);
            };

            for (int j = 0; j != O; ++j)
                _b[j] = quantize(b[j]);

            int16_t* const rows = &_w[0][0];
            for (size_t i = 0; i != size_t(I) * O; ++i)
                rows[i] = quantize(w[i]);
        }
    };


    /* int8 layer vectors: int16 accumulator lanes, u8 activations, s8 weights, int32 sums */
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


    /* sums[k] = in . wt[k] for R rows, int32 */
    template <int R, int N>
    INLINE void dot_rows(const uint8_t (&in)[N], const int8_t (*wt)[N], int32_t (&sums)[R])
    {
        static_assert(N % VecU8::size() == 0);

        VecI32 acc[R];
        for (int k = 0; k != R; ++k)
            acc[k] = VecI32(0);

        VecU8 a;
        VecS8 w;
        for (int i = 0; i != N; i += VecU8::size())
        {
            a.load_a(&in[i]);
            for (int k = 0; k != R; ++k)
            {
                w.load_a(&wt[k][i]);
                acc[k] = dot_add(acc[k], a, w);
            }
        }

        for (int k = 0; k != R; ++k)
            sums[k] = ::horizontal_add(acc[k]);
    }


    /* int16 accumulator at QSCALE -> u8 activations: clamp to [0, ACT_CLAMP], >> ACT_SHIFT */
    template <int N>
    INLINE void activate(const int16_t (&in)[N], uint8_t (&out)[N])
    {
        constexpr int W = VecS16::size();
        static_assert(2 * W == VecU8::size() && N % (2 * W) == 0);

        const VecS16 lo(0), hi(ACT_CLAMP);
        for (int i = 0; i != N; i += 2 * W)
        {
            const VecS16 a = min(max(VecS16().load_a(&in[i]), lo), hi) >> ACT_SHIFT;
            const VecS16 b = min(max(VecS16().load_a(&in[i + W]), lo), hi) >> ACT_SHIFT;
            /* values are in [0, ACT_MAX], so the signed saturating pack is exact */
            VecU8(compress_saturated(a, b)).store_a(&out[i]);
        }
    }


    /* u8 activations x s8 weights at L2_WSCALE; int32 sums + bias at L2_SCALE, relu, to float */
    template <int I, int O>
    struct DenseInt8Layer
    {
        static constexpr int INPUTS = I;
        static constexpr int OUTPUTS = O;

        ALIGN int32_t _b[OUTPUTS];
        ALIGN int8_t _wt[OUTPUTS][INPUTS];

        static constexpr size_t param_count()
        {
            return (I + 1) * O;
        }

        void load_weights(std::istream& file)
        {
            auto w = std::make_unique<float[]>(I * O);
            auto b = std::make_unique<float[]>(O);

            file.read(reinterpret_cast<char*>(w.get()), I * O * sizeof(float));
            file.read(reinterpret_cast<char*>(b.get()), O * sizeof(float));

            for (int j = 0; j != O; ++j)
            {
                const auto q = std::round(double(b[j]) * L2_SCALE);
                if (q > INT32_MAX || q < INT32_MIN)
                    throw std::runtime_error("bias " + std::to_string(b[j]) + " exceeds int32 at scale " + std::to_string(L2_SCALE));
                _b[j] = int32_t(q);
            }

            for (int i = 0; i != I; ++i)
                for (int j = 0; j != O; ++j)
                {
                    const auto q = std::round(w[i * O + j] * L2_WSCALE);
                    if (std::abs(q) > INT8_MAX) /* symmetric: -128 rejected */
                        throw std::runtime_error("weight " + std::to_string(w[i * O + j]) + " exceeds int8 at scale " + std::to_string(L2_WSCALE));
                    _wt[j][i] = int8_t(q);
                }
        }

        INLINE void dot(const uint8_t (&input)[INPUTS], float (&output)[OUTPUTS]) const
        {
            constexpr int R = 4;
            static_assert(OUTPUTS % R == 0);
            constexpr float OUT_SCALE = 1.0f / L2_SCALE;

            for (int j = 0; j != OUTPUTS; j += R)
            {
                int32_t sums[R];
                dot_rows<R>(input, &_wt[j], sums);

                for (int k = 0; k != R; ++k)
                    output[j + k] = float(std::max(0, sums[k] + _b[j + k])) * OUT_SCALE;
            }
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

        struct Bucket
        {
            ALIGN int16_t output[OUTPUTS] = { };
            uint64_t hash = 0;
        };

        /* Per-thread cache: last accumulator computed in each bucket, plus its
         * piece bitboards, so bucket changes refresh by diffing pieces against
         * a same-bucket position instead of rebuilding all active inputs.
         */
        struct RefreshEntry
        {
            ALIGN int16_t output[OUTPUTS];
            Bitboard pieces[6][2] = { };
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

        INLINE Bucket& slot(int bucket) { return _bucket[NNUE_SINGLE_BUCKET ? 0 : bucket]; }
        INLINE const Bucket& slot(int bucket) const { return _bucket[NNUE_SINGLE_BUCKET ? 0 : bucket]; }

        INLINE const int16_t (&output() const)[OUTPUTS] { return slot(_current_bucket).output; }

    #if DEBUG_INCREMENTAL
        /* remember previous inputs, for debugging */
        ALIGN input_t _input[ACTIVE_INPUTS] = { }; /* one-hot encoding */

        /* full-width shadow of the per-bucket hashes, so the bucket-vs-hash
         * equivalence can be checked even when SLOTS == 1 (single-bucket mode)
         */
        uint64_t _ref_hash[NUM_BUCKETS] = { };
    #endif /* DEBUG_INCREMENTAL */


        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != slot(_current_bucket).hash;
        }


        /** out = src - rows(remove) + rows(add), both perspectives; with Copy, out2 gets a copy too.
         * Indices are white-view: the black half uses the mirrored bucket and index ^ PERSPECTIVE_XOR.
         */
        template <bool Copy = false, typename LA>
        static INLINE void apply_deltas(
            const LA& layer,
            int bucket,
            const int16_t* src,
            int16_t* out,
            int16_t* out2,
            const int* remove,
            int r_idx,
            const int* add,
            int a_idx)
        {
            static_assert(LA::OUTPUTS == HALF);

            const size_t base_w = size_t(bucket) * ACTIVE_INPUTS;
            const size_t base_b = size_t(mirror_bucket(bucket)) * ACTIVE_INPUTS;

            VecShort vb, vw, v;
            for (int j = 0; j != HALF; j += VecShort::size())
            {
                vb.load_a(&src[j]);
                vw.load_a(&src[HALF + j]);

                for (int i = 0; i < r_idx; ++i)
                {
                    v.load_a(&layer._w[base_b + (remove[i] ^ PERSPECTIVE_XOR)][j]);
                    vb -= v;
                    v.load_a(&layer._w[base_w + remove[i]][j]);
                    vw -= v;
                }

                for (int i = 0; i < a_idx; ++i)
                {
                    v.load_a(&layer._w[base_b + (add[i] ^ PERSPECTIVE_XOR)][j]);
                    vb += v;
                    v.load_a(&layer._w[base_w + add[i]][j]);
                    vw += v;
                }

                vb.store_a(&out[j]);
                vw.store_a(&out[HALF + j]);
                if constexpr (Copy)
                {
                    vb.store_a(&out2[j]);
                    vw.store_a(&out2[HALF + j]);
                }
            }
        }


        /** Compute 1st layer output from scratch at root */
        template <typename LA>
        INLINE void full_update(const LA& layer, const State& state, int bucket)
        {
            int add_inputs[MAX_ACTIVE_INPUTS];
            int a_idx = 0;
            for_each_piece_input(state, [&](int i) { add_inputs[a_idx++] = i; });

            ALIGN int16_t bias[OUTPUTS];
            memcpy(bias, layer._b, sizeof(layer._b));
            memcpy(bias + HALF, layer._b, sizeof(layer._b));

            apply_deltas(layer, bucket, bias, slot(bucket).output, nullptr, nullptr, 0, add_inputs, a_idx);

            slot(bucket).hash = state.hash();
            _current_bucket = bucket;
        #if DEBUG_INCREMENTAL
            memset(&_input, 0, sizeof(_input));
            one_hot_encode(state, _input);
            _ref_hash[bucket] = state.hash();
        #endif
        }

        template <typename LA>
        INLINE void update(const LA& layer, const State& state)
        {
            if (needs_update(state))
            {
                full_update(layer, state, get_bucket(state));
            }
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
            ASSERT(ancestor.slot(ancestor._current_bucket).hash == prev.hash());

            const int bucket = get_bucket(state);
            const int prev_bucket = ancestor._current_bucket;
            const bool incremental = (bucket == prev_bucket);

        #if DEBUG_INCREMENTAL
            /* bucket==prev_bucket must match the per-bucket-hash predicate; _ref_hash shadows it for SLOTS==1 */
            ASSERT_ALWAYS(incremental == (ancestor._ref_hash[bucket] == prev.hash()));
          #if !NNUE_SINGLE_BUCKET
            ASSERT_ALWAYS(ancestor._ref_hash[bucket] == ancestor._bucket[bucket].hash);
          #endif /* !NNUE_SINGLE_BUCKET */
            {
                static std::atomic<bool> seen[NUM_BUCKETS][NUM_BUCKETS];
                if (!seen[prev_bucket][bucket].exchange(true))
                    std::cerr << "[nnue] bucket " << prev_bucket << " -> " << bucket << (bucket == prev_bucket ? " same" : " cross") << std::endl;
            }
        #endif /* DEBUG_INCREMENTAL */

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

            if (incremental)
            {
                apply_deltas(layer, bucket, ancestor.slot(bucket).output, slot(bucket).output, nullptr,
                    remove_inputs, r_idx, add_inputs, a_idx);
            }
            else if (slot(bucket).hash != state.hash())
            {
                auto& entry = refresh.entry[bucket];

                if (!entry.valid)
                {
                    /* empty-board entry: the diff below rebuilds from bias + all pieces */
                    memcpy(entry.output, layer._b, sizeof(layer._b));
                    memcpy(entry.output + HALF, layer._b, sizeof(layer._b));
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

                apply_deltas<true>(layer, bucket, entry.output, entry.output, slot(bucket).output, rem, nr, add, na);
            }

            slot(bucket).hash = state.hash();
            _current_bucket = bucket;

        #if DEBUG_INCREMENTAL
            _ref_hash[bucket] = state.hash();

            // Validate the incremental result against a full recompute
            Accumulator full;
            full.full_update(layer, state, bucket);
            for (int i = 0; i != OUTPUTS; ++i)
                ASSERT_ALWAYS(full.slot(bucket).output[i] == slot(bucket).output[i]);
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
        static_assert(A::OUTPUTS == L2::INPUTS);
        static_assert(L2::OUTPUTS == L3::INPUTS);
        static_assert(L3::OUTPUTS == OUT::INPUTS);

        ALIGN uint8_t l2_in[A::OUTPUTS];
        ALIGN float l2_out[L2::OUTPUTS];
        ALIGN float l3_out[L3::OUTPUTS];
        ALIGN float output[1]; // eval

        activate(a.output(), l2_in);
        l2.dot(l2_in, l2_out);
        l3.dot(l2_out, l3_out, [](const Vector& v) { return relu(v); });
        out.dot(l3_out, output);
        return EVAL_SCALE * output[0];
    }


    /* Compute the move head's sub-accumulator once per node: relu(bias + W_acc . active),
     * kept in the quantized int16 domain (like the eval accumulator, no dequantize). Sparse
     * gather over active inputs; the result feeds score_move for every move at this node.
     * LMA is the [MOVE_INPUTS x MOVE_ACC] quantized layer.
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
