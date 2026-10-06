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
    constexpr int MAX_ACTIVE_INPUTS = 32; /* one per piece */
    constexpr int NUM_BUCKETS = 16;
    constexpr int PAWN_BUCKETS = chess::PAWN_BUCKETS;
    constexpr int KING_BUCKETS = 4;
    static_assert(NUM_BUCKETS == PAWN_BUCKETS * KING_BUCKETS, "bucket grid mismatch");
    constexpr int QSCALE = 1024;

    /* black-view input index: color swap (64) + rank flip (56) */
    constexpr int PERSPECTIVE_XOR = 120;

    /* accumulator rows: 32 pieces + bias must not overflow int16 */
    constexpr int ACC_WEIGHT_MAX = INT16_MAX / (MAX_ACTIVE_INPUTS + 1);

    /* Activation: cap the accumulator to [0, 1023], divide by 8, giving bytes 0..127 (1.0 == 128) */
    /* 1024 levels become 128: the int8 dot product needs bytes, so trade precision for throughput */

    constexpr int ACT_SHIFT = 3;
    constexpr int ACT_MAX = 127;
    constexpr int ACT_CLAMP = ((ACT_MAX + 1) << ACT_SHIFT) - 1;
    constexpr int ACT_SCALE = QSCALE >> ACT_SHIFT;

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

    /* weight rows per pass: enough independent sums that the CPU never waits on a dot_add */
    constexpr int DOT_ROWS = (INSTRSET >= 10) ? 16 : 8; /* 16 needs the 32 registers of AVX-512 */


    /* Dot products of the input with R weight rows at once.
     * The rows are stored interleaved, chunk by chunk, so that
     * we can read them as one sequential stream.
     */
    template <int R, int N>
    INLINE void dot_rows(const uint8_t (&in)[N], const int8_t* rows, int32_t (&sums)[R])
    {
        constexpr int W = VecU8::size();
        static_assert(N % W == 0 && R % 8 == 0);

        VecI32 acc[R];
        for (int k = 0; k != R; ++k)
            acc[k] = VecI32(0);

        VecU8 a;
        VecS8 w;
        for (int i = 0; i != N; i += W)
        {
            a.load_a(&in[i]);
            for (int k = 0; k != R; ++k)
            {
                w.load_a(rows);
                rows += W;
                acc[k] = dot_add(acc[k], a, w);
            }
        }

        /* sum up 8 vectors at a time */
        for (int k = 0; k != R; k += 8)
            ::horizontal_add(*reinterpret_cast<const VecI32 (*)[8]>(&acc[k]), *reinterpret_cast<int32_t (*)[8]>(&sums[k]));
    }


    /* Clipped ReLU: cap the int16 accumulator to [0, 1023] and shrink it to bytes 0..127 */
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
        ALIGN int32_t _b[O]; /* biases */
        ALIGN int8_t _packed[O * INPUTS]; /* weights in the order dot() reads them, see Layer::packed */
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

        /* int8 layers store the weights in the order dot() reads them.
         *
         *  as trained: one row of 2048 bytes per output, read in vector-sized chunks (c0, c1, ... 64 bytes each on AVX-512)
         *      row 0:  [ c0 | c1 | c2 | ... ]
         *      row 1:  [ c0 | c1 | c2 | ... ]
         *      ...
         *  as stored: DOT_ROWS rows at a time, chunk by chunk
         *      [ row0.c0  row1.c0 ... row15.c0 ][ row0.c1  row1.c1 ... row15.c1 ][ ... ]
         *
         * dot_rows takes one chunk of activations and multiplies it with the same chunk of DOT_ROWS rows,
         * then moves to the next chunk. In the trained layout those chunks sit 2KB apart, one stream per
         * row for the hardware prefetcher to follow; in the stored layout they are adjacent, so a pass
         * reads memory front to back as one stream.
         */
        static constexpr bool PACKED = std::is_same_v<T, int8_t> && !Incremental;

        INLINE T* packed(int j, int i)
        {
            constexpr int W = VecU8::size();
            const int group = j / DOT_ROWS, chunk = i / W;
            return &this->_packed[((size_t(group) * (INPUTS / W) + chunk) * DOT_ROWS + j % DOT_ROWS) * W + i % W];
        }

        INLINE const T* packed(int j, int i) const
        {
            return const_cast<Layer*>(this)->packed(j, i);
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
                        *packed(j, i) = v;
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
                            *packed(j, i) = 0;
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

        /** Byte activations times int8 weights, summed in int32, then relu and back to float */
        INLINE void dot(const uint8_t (&input)[INPUTS], float (&output)[OUTPUTS]) const
        {
            static_assert(PACKED);

            constexpr int R = DOT_ROWS;
            static_assert(OUTPUTS % R == 0);
            constexpr float OUT_SCALE = 1.0f / BIAS_SCALE;

            ALIGN int32_t sums[OUTPUTS / R][R];
            for (int j = 0; j != OUTPUTS; j += R)
                dot_rows(input, packed(j, 0), sums[j / R]);

            for (int j = 0; j != OUTPUTS; ++j)
                output[j] = float(std::max(0, sums[j / R][j % R] + this->_b[j])) * OUT_SCALE;
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

        /* Update both perspectives: subtract the rows of removed pieces, add the new ones; with Copy, out2 gets a copy too.
         * REMOVED and ADDED fix the piece counts at compile time, which turns the loops into straight-line code.
         *
         *  piece i (white-view index) in bucket b:
         *    white half  +=  layer._w[ b         * 768 + i       ]   (1024 values)
         *    black half  +=  layer._w[ mirror(b) * 768 + (i^120) ]   (1024 values)
         *                              ^ king bits swapped ^ color swap (64) + rank flip (56)
         */
        template <int REMOVED = DYNAMIC, int ADDED = DYNAMIC, bool Copy = false, typename LA>
        static INLINE void apply_deltas(
            const LA& layer,
            int bucket,
            const int16_t (&src)[OUTPUTS],
            int16_t (&out)[OUTPUTS],
            int16_t (&out2)[OUTPUTS],
            const int (&remove)[MAX_ACTIVE_INPUTS],
            int r_idx,
            const int (&add)[MAX_ACTIVE_INPUTS],
            int a_idx)
        {
            static_assert(LA::OUTPUTS == HALF);

            const int removed = (REMOVED == DYNAMIC) ? r_idx : REMOVED;
            const int added = (ADDED == DYNAMIC) ? a_idx : ADDED;
            ASSERT(removed == r_idx && added == a_idx);

            const size_t base_w = size_t(bucket) * ACTIVE_INPUTS;
            const size_t base_b = size_t(mirror_bucket(bucket)) * ACTIVE_INPUTS;

            VecShort vb, vw, v;
            for (int j = 0; j != HALF; j += VecShort::size())
            {
                vb.load_a(&src[j]);
                vw.load_a(&src[HALF + j]);

                for (int i = 0; i < removed; ++i)
                {
                    v.load_a(&layer._w[base_b + (remove[i] ^ PERSPECTIVE_XOR)][j]);
                    vb -= v;
                    v.load_a(&layer._w[base_w + remove[i]][j]);
                    vw -= v;
                }

                for (int i = 0; i < added; ++i)
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


        /* Update after a move; the usual moves get their own compile-time version of apply_deltas */
        template <typename LA>
        static INLINE void apply_move(
            const LA& layer,
            int bucket,
            const int16_t (&src)[OUTPUTS],
            int16_t (&out)[OUTPUTS],
            const int (&remove)[MAX_ACTIVE_INPUTS],
            int removed,
            const int (&add)[MAX_ACTIVE_INPUTS],
            int added)
        {
            /* out2 is unused without Copy */
            if (removed == 1 && added == 1) /* quiet move or promotion */
                apply_deltas<1, 1>(layer, bucket, src, out, out, remove, removed, add, added);
            else if (removed == 2 && added == 1) /* capture */
                apply_deltas<2, 1>(layer, bucket, src, out, out, remove, removed, add, added);
            else if (removed == 2 && added == 2) /* castling */
                apply_deltas<2, 2>(layer, bucket, src, out, out, remove, removed, add, added);
            else
                apply_deltas(layer, bucket, src, out, out, remove, removed, add, added);
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

            auto& out = slot(bucket).output;
            apply_deltas(layer, bucket, bias, out, out, NONE, 0, add_inputs, a_idx);

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
                apply_move(layer, bucket, ancestor.slot(bucket).output, slot(bucket).output, remove_inputs, r_idx, add_inputs, a_idx);
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

                apply_deltas<DYNAMIC, DYNAMIC, true>(layer, bucket, entry.output, entry.output, slot(bucket).output, rem, nr, add, na);
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
} /* namespace nnue */
