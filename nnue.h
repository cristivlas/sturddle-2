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

#define DEBUG_INCREMENTAL false


namespace nnue
{
    using namespace chess;
    using input_t = int16_t;

    constexpr int ACTIVE_INPUTS = 897;
    constexpr int EVAL_SCALE = 100;

    constexpr auto POOL_STRIDE = Vec8s::size();
    constexpr int QSCALE = 1024;

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
    // static const Vec16s v16_zero(0);


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
        for (const auto bb : {
            board.kings, board.pawns, board.knights, board.bishops, board.rooks, board.queens })
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

    template <>
    INLINE Vector relu<Vector>(Vector v) { return max(v, v_zero); }

    template <>
    INLINE Vec8s relu<Vec8s>(Vec8s v) { return max(v, v8_zero); }

    // template <>
    // INLINE Vec16s relu<Vec16s>(Vec16s v) { return max(v, v16_zero); }

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

            constexpr auto MAX_INPUT = std::min<int>(INPUTS, INPUT_SIZE);

            for (int j = 0; j != OUTPUTS; j += N)
            {
                #pragma unroll N
                for (int k = 0; k != N; ++k)
                    sum[k] = VSum(0);

                for (int i = 0; i < MAX_INPUT; i += N)
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

            out[j] = float(::horizontal_add(extend(v))) / POOL_STRIDE / QSCALE;
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

            out[j] = float(::horizontal_add(extend(max(v.get_low(), v8_zero)))) / POOL_STRIDE / QSCALE;
            out[j + 1] = float(::horizontal_add(extend(max(v.get_high(), v8_zero)))) / POOL_STRIDE / QSCALE;
        }
    #endif /* INSTRSET < 8 */
    }


    template <int M, int N, int O> struct Accumulator
    {
        static_assert(ACTIVE_INPUTS * 4 == M);

        static constexpr int INPUTS = round_up<INPUT_STRIDE>(M);
        static constexpr int OUTPUTS_A = N;
        static constexpr int OUTPUTS_B = O;

        static constexpr size_t MAX_DELTA = 32; /* total pieces */
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
    #if USE_MOVE_PREDICTION
        ALIGN int16_t _move_logits[4096] = { };
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
    #if USE_MOVE_PREDICTION
        template <typename LA, typename LB, typename LM>
        INLINE void full_update(const LA& layer_1a, const LB& layer_1b, const LM& layer_m, const State& state, int bucket)
    #else
        template <typename LA, typename LB>
        INLINE void full_update(const LA& layer_1a, const LB& layer_1b, const State& state, int bucket)
    #endif
        {
        #if DEBUG_INCREMENTAL
            memset(&_input, 0, sizeof(_input));
        #else
            ALIGN input_t _input[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)] = { };
        #endif /* DEBUG_INCREMENTAL */

            one_hot_encode(state, _input);

            const size_t base = bucket * ACTIVE_INPUTS;

            layer_1a.dot(_input, _bucket[bucket].output, base);
            layer_1b.dot(_input, _output_b);

        #if USE_MOVE_PREDICTION
            layer_m.dot(_input, _move_logits);
        #endif

            _bucket[bucket].hash = state.hash();
            _current_bucket = bucket;
        }

    #if USE_MOVE_PREDICTION
        template <typename LA, typename LB, typename LM>
        INLINE void update(const LA& layer_1a, const LB& layer_1b, const LM& layer_m, const State& state)
        {
            if (needs_update(state))
            {
                full_update(layer_1a, layer_1b, layer_m, state, get_bucket(state));
            }
        }
    #else
        template <typename LA, typename LB>
        INLINE void update(const LA& layer_1a, const LB& layer_1b, const State& state)
        {
            if (needs_update(state))
            {
                full_update(layer_1a, layer_1b, state, get_bucket(state));
            }
        }
    #endif /* USE_MOVE_PREDICTION */

        /** Utility for incremental updates */
        static INLINE void delta(int (&d)[MAX_DELTA], int& idx, PieceType pt, Color col, Square sq)
        {
            d[idx++] = piece_square_index(pt, col, sq);
            d[idx++] = mask_index(col, sq);
        }

        /** Update 1st layer output incrementally, based on a previous state */
    #if USE_MOVE_PREDICTION
        template <typename LA, typename LB, typename LM, typename A>
        INLINE void update(
            const LA& layer_a,
            const LB& layer_b,
            const LM& layer_m,
            const State& prev,
            const State& state,
            const Move& move,
            A& ancestor)
    #else
        template <typename LA, typename LB, typename A>
        INLINE void update(
            const LA& layer_a,
            const LB& layer_b,
            const State& prev,
            const State& state,
            const Move& move,
            A& ancestor)
    #endif /* USE_MOVE_PREDICTION */
        {
            ASSERT(needs_update(state));

            const int bucket = get_bucket(state);
            const bool can_incremental_a = (ancestor._bucket[bucket].hash == prev.hash());

            /* compute delta based on ancestor state */
            ASSERT(prev.turn != state.turn);

            int remove_inputs[MAX_DELTA];
            int add_inputs[MAX_DELTA];
            int r_idx = 0, a_idx = 0;

            if (move)
            {
                get_deltas(prev, state, move, prev.turn, remove_inputs, add_inputs, r_idx, a_idx);

                ASSERT(a_idx < MAX_DELTA);
                ASSERT(r_idx < MAX_DELTA);
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

            if (can_incremental_a)
            {
                memcpy(_bucket[bucket].output, ancestor._bucket[bucket].output, sizeof(_bucket[bucket].output));
            }
            else if (_bucket[bucket].hash != state.hash())
            {
                /* Full update for layer A only */
            #if DEBUG_INCREMENTAL
                layer_a.dot(_input, _bucket[bucket].output, base);
            #else
                ALIGN input_t _input[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)] = { };
                one_hot_encode(state, _input);
                layer_a.dot(_input, _bucket[bucket].output, base);
            #endif
            }

            /* Layer B and M: always incremental from ancestor */
            memcpy(_output_b, ancestor._output_b, sizeof(_output_b));
        #if USE_MOVE_PREDICTION
            memcpy(_move_logits, ancestor._move_logits, sizeof(_move_logits));
        #endif
        #if USE_MOVE_PREDICTION
            incremental_update(layer_a, layer_b, layer_m, remove_inputs, add_inputs, r_idx, a_idx, base, bucket, can_incremental_a);
        #else
            incremental_update(layer_a, layer_b, remove_inputs, add_inputs, r_idx, a_idx, base, bucket, can_incremental_a);
        #endif
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
    #if USE_MOVE_PREDICTION
        template <typename LA, typename LB, typename LM>
        INLINE void incremental_update(
            const LA& layer_a,
            const LB& layer_b,
            const LM& layer_m,
            const int (&remove_inputs)[MAX_DELTA],
            const int (&add_inputs)[MAX_DELTA],
            const int r_idx,
            const int a_idx,
            size_t base,
            int bucket,
            bool update_layer_a)
    #else
        template <typename LA, typename LB>
        INLINE void incremental_update(
            const LA& layer_a,
            const LB& layer_b,
            const int (&remove_inputs)[MAX_DELTA],
            const int (&add_inputs)[MAX_DELTA],
            const int r_idx,
            const int a_idx,
            size_t base,
            int bucket,
            bool update_layer_a)
    #endif
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

        #if USE_MOVE_PREDICTION
            for (int j = 0; j != 4096; j += VecShort::size())
            {
                vo.load_a(&_move_logits[j]);

                for (int i = 0; i < r_idx; ++i)
                {
                    const auto index = remove_inputs[i];
                    ASSERT(index < LM::INPUTS);
                    vw.load_a(&layer_m._w[index][j]);
                    vo -= vw;
                }

                for (int i = 0; i < a_idx; ++i)
                {
                    const auto index = add_inputs[i];
                    ASSERT(index < LM::INPUTS);
                    vw.load_a(&layer_m._w[index][j]);
                    vo += vw;
                }
                vo.store_a(&_move_logits[j]);
            }
        #endif /* USE_MOVE_PREDICTION */

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
            int (&remove)[MAX_DELTA],
            int (&add)[MAX_DELTA],
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

        ALIGN float attn_in[ATTN::INPUTS];
        ALIGN float attn_out[ATTN::OUTPUTS];
        ALIGN float l2_in[L2::INPUTS];
        ALIGN float l2_out[L2::OUTPUTS];
        ALIGN float l3_out[L3::OUTPUTS];
        ALIGN float output[1]; // eval

        pool(a._bucket[a._current_bucket].output, l2_in);

        /* The "spatial attention" layer modulates L2. */
        activate(a._output_b, attn_in);
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

        out.dot(l3_out, output);
        return EVAL_SCALE * output[0];
    }
} /* namespace nnue */
