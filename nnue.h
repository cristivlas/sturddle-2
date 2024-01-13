#pragma once
/*
 * Sturddle Chess Engine (C) 2023 Cristian Vlasceanu
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

#if (__amd64__) || (__x86_64__) || (__i386__) || (_M_AMD64) || (_M_X64) || (_M_IX86)
    #include "vectorclass.h"
#elif (__arm__) || (__arm64__) || (__aarch64__)
    #include "armvector.h"
#endif

#if INSTRSET >= 9 /* AVX 512 */
    #define ALIGN alignas(64)
    #ifndef ARCH
        #define ARCH "AVX512"
    #endif
#else
    #define ALIGN alignas(32)
    #ifndef ARCH
        #if INSTRSET >= 8
            #define ARCH "AVX2"
        #else
            #define ARCH "SSE2"
        #endif
    #endif /* ARCH */
#endif /* INSTRSET >= 9 */


#define DEBUG_INCREMENTAL false


namespace nnue
{
    using namespace chess;
    using input_t = int16_t;

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
    #elif INSTRSET >= 8
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
    static const std::string instrset = ARCH "/FMA";
#else
    static const std::string instrset = ARCH;
#endif /* __FMA__ */

    INLINE Vec16s horizontal_add(const Vec16s (&v)[16])
    {
        return Vec16s(
            horizontal_add(v[0]), horizontal_add(v[1]), horizontal_add(v[2]), horizontal_add(v[3]),
            horizontal_add(v[4]), horizontal_add(v[5]), horizontal_add(v[6]), horizontal_add(v[7]),
            horizontal_add(v[8]), horizontal_add(v[9]), horizontal_add(v[10]),horizontal_add(v[11]),
            horizontal_add(v[12]),horizontal_add(v[13]),horizontal_add(v[14]),horizontal_add(v[15]));
    }

    INLINE Vector horizontal_add(const Vector (&v)[1])
    {
        return horizontal_add(v[0]);
    }

    INLINE bool all_zero(const Vec16s& v)
    {
        return !horizontal_or(v);
    }

    template <int N> INLINE void load_partial(Vector& v, const float* p)
    {
        if constexpr (N == 1)
            #if INSTRSET >= 9
                v.load_partial(1, p);
            #elif INSTRSET >= 8
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
            #if INSTRSET >= 9
                v.store_partial(1, p);
            #elif INSTRSET >= 8
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

    template <typename T>
    INLINE void one_hot_encode(const State& board, T (&encoding)[897])
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
    template <int N>
    INLINE void activation(const int16_t (&input)[N], float (&output)[N])
    {
        constexpr float QSCALE_RECIP = 1.0f / QSCALE;

        #pragma clang loop vectorize(enable)
        for (int i = 0; i != N; ++i)
            output[i] = std::max<float>(0, float(input[i]) * QSCALE_RECIP);
    }


    template <int I, int O, typename T=float, int Scale=1>
    struct Layer
    {
        static constexpr int INPUTS = I;
        static constexpr int OUTPUTS = O;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _w[INPUTS][OUTPUTS]; /* weights */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */

        Layer(const float(&w)[INPUTS][OUTPUTS], const float(&b)[OUTPUTS])
        {
            set_weights(w, b);
        }

        void set_weights(const float(&w)[INPUTS][OUTPUTS], const float(&b)[OUTPUTS])
        {
            for (int j = 0; j != OUTPUTS; ++j)
                _b[j] = b[j] * Scale;

            for (int i = 0; i != INPUTS; ++i)
                for (int j = 0; j != OUTPUTS; ++j)
                    _w[i][j] = _wt[j][i] = w[i][j] * Scale;
        }

        void set_weights(const std::vector<std::vector<float>>& w, const std::vector<float>& b)
        {
            if (w.size() != INPUTS || w[0].size() != OUTPUTS || b.size() != OUTPUTS)
                throw std::invalid_argument("Input dimensions do not match layer dimensions");

            float weights[INPUTS][OUTPUTS];
            float biases[OUTPUTS];

            for (int i = 0; i < INPUTS; ++i)
                for (int j = 0; j < OUTPUTS; ++j)
                    weights[i][j] = w[i][j];

            for (int j = 0; j < OUTPUTS; ++j)
                biases[j] = b[j];

            set_weights(weights, biases);
        }


        /* input */
        template <size_t S, typename F>
        static INLINE void dot(
            const input_t (&input)[S],
            int16_t (&output)[OUTPUTS],
            const int16_t(&b)[OUTPUTS],
            const int16_t(&w)[INPUTS][OUTPUTS],
            const int16_t(&wt)[OUTPUTS][INPUTS],
            F /* activation applied separately */
        )
        {
            static_assert(S >= INPUTS);

        #if 0 /* testing */
            for (int j = 0; j != OUTPUTS; ++j)
            {
                output[j] = b[j];
                #pragma clang loop vectorize(enable)
                for (int i = 0; i != INPUTS; ++i)
                    output[j] += input[i] * wt[j][i];
            }
        #else
            constexpr auto N = Vec16s::size();
            static_assert(OUTPUTS % N == 0);

            constexpr auto R = round_down<N>(INPUTS);

            Vec16s in, vw, sum[N];

            for (int j = 0; j != OUTPUTS; j += N)
            {
                #pragma unroll N
                for (int k = 0; k != N; ++k)
                    sum[k] = Vec16s(0);

                for (int i = 0; i != R; i += N)
                {
                    in.load_a(input + i);
                    if (all_zero(in))
                        continue;

                    for (int k = 0; k != N; ++k)
                    {
                        vw.load(&wt[j + k][i]);
                        sum[k] += in * vw;
                    }
                }

                auto sums = horizontal_add(sum);

                for (int i = R; i != INPUTS; ++i)
                {
                    vw.load_a(&w[i][j]);
                    sums += input[i] * vw;
                }
                vw.load_a(&b[j]);
                (vw + sums).store_a(&output[j]);
            }
        #endif
        }


        /* hidden, output */
        template <typename F>
        static INLINE void dot(
            const float (&input)[INPUTS],
            float (&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&)[INPUTS][OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS],
            F activate
        )
        {
            constexpr int N = Vector::size();
            constexpr int Q = (OUTPUTS % N == 0) ? N : 1;

            static_assert(INPUTS % N == 0);

            Vector sum[Q], v_wt, v_in, v_out;

            for (int j = 0; j != OUTPUTS; j += Q)
            {
                #pragma unroll Q
                for (int k = 0; k != Q; ++k)
                    sum[k] = Vector(0.0);

                for (int i = 0; i != INPUTS; i += N)
                {
                    v_in.load_a(&input[i]);
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
            dot(input, output, _b, _w, _wt, [](const Vector& v) { return v; });
        }

        template <size_t N, typename U, typename V, typename F>
        INLINE void dot(const U (&input)[N], V (&output)[OUTPUTS], F activate) const
        {
            dot(input, output, _b, _w, _wt, activate);
        }
    };


    template <size_t INPUTS, size_t OUTPUTS>
    INLINE void pool(const float (&in)[INPUTS], float (&out)[OUTPUTS])
    {
        constexpr size_t stride = INPUTS / OUTPUTS;

        static_assert(INPUTS % OUTPUTS == 0);
        static_assert(stride == 4);

        Vec4f v;

        for (size_t i = 0, j = 0; i + stride <= INPUTS; i += stride, ++j)
        {
            v.load_a(&in[i]);
            out[j] = horizontal_add(v) / Vec4f::size();
        }
    }


    template <int M, int N, int O> struct Accumulator
    {
        static constexpr int INPUTS = M;
        static constexpr int OUTPUTS_A = N;
        static constexpr int OUTPUTS_B = O;

    #if DEBUG_INCREMENTAL
        ALIGN input_t _input[INPUTS] = { 0 }; /* one-hot encoding */
    #endif
        ALIGN int16_t _output_a[OUTPUTS_A] = { 0 };
        ALIGN int16_t _output_b[OUTPUTS_B] = { 0 };
        uint64_t _hash = 0;

        /** Compute 1st layer output from scratch at root */
        template <typename LA, typename LB>
        INLINE void update(const LA& layer_1a, const LB& layer_1b, const State& state)
        {
            if (needs_update(state))
            {
                _hash = state.hash();

            #if DEBUG_INCREMENTAL
                memset(&_input, 0, sizeof(_input));
            #else
                ALIGN input_t _input[INPUTS] = { 0 };
            #endif
                one_hot_encode(state, _input);

                layer_1a.dot(_input, _output_a);
                layer_1b.dot(_input, _output_b);
            }
        }

        /** Utility for incremental updates */
        static INLINE void delta(int (&d)[INPUTS], int& idx, PieceType pt, Color col, Square sq)
        {
            d[idx++] = piece_square_index(pt, col, sq);
            d[idx++] = mask_index(col, sq);
        }

        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != _hash;
        }

        /** Update 1st layer output incrementally, based on a previous state */
        template <typename LA, typename LB, typename A>
        INLINE void update(
            const LA& layer_a,
            const LB& layer_b,
            const State& prev,
            const State& state,
            const Move& move,
            A& ancestor)
        {
            if (needs_update(state))
            {
                _hash = state.hash();

                /* compute delta based on ancestor state */
                ASSERT(prev.turn != state.turn);

                memcpy(_output_a, ancestor._output_a, sizeof(_output_a));
                memcpy(_output_b, ancestor._output_b, sizeof(_output_b));

            #if DEBUG_INCREMENTAL
                memcpy(_input, ancestor._input, sizeof(_input));
            #endif
                int remove_inputs[INPUTS];
                int add_inputs[INPUTS];
                int r_idx = 0, a_idx = 0;

                if (move)
                {
                    update(prev, state, move, prev.turn, remove_inputs, add_inputs, r_idx, a_idx);

                    ASSERT(a_idx < INPUTS);
                    ASSERT(r_idx < INPUTS);
                }

            #if DEBUG_INCREMENTAL
                for (int i = 0; i != r_idx; ++i)
                    _input[remove_inputs[i]] = 0;
                for (int i = 0; i != a_idx; ++i)
                    _input[add_inputs[i]] = 1;

                _input[TURN_INDEX] ^= 1;

                input_t temp[INPUTS] = { 0 };
                one_hot_encode(state, temp);

                for (int i = 0; i != INPUTS; ++i)
                    ASSERT_ALWAYS(_input[i] == temp[i]);

            #endif /* DEBUG_INCREMENTAL */

                if (state.turn)
                    add_inputs[a_idx++] = TURN_INDEX;
                else
                    remove_inputs[r_idx++] = TURN_INDEX;

                recompute(layer_a, layer_b, remove_inputs, add_inputs, r_idx, a_idx);

            #if DEBUG_INCREMENTAL
                int16_t output_a[OUTPUTS_A] = { 0 };
                layer_a.dot(_input, output_a);
                for (int i = 0; i != OUTPUTS_A; ++i)
                    ASSERT_ALWAYS(abs(output_a[i] - _output_a[i]) < 0.0001);

                int16_t output_b[OUTPUTS_B] = { 0 };
                layer_b.dot(_input, output_b);
                for (int i = 0; i != OUTPUTS_B; ++i)
                    ASSERT_ALWAYS(abs(output_b[i] - _output_b[i]) < 0.0001);
            #endif /* DEBUG_INCREMENTAL */
            }
        }

        template <typename LA, typename LB>
        INLINE void recompute(
            const LA& layer_a,
            const LB& layer_b,
            const int (&remove_inputs)[INPUTS],
            const int (&add_inputs)[INPUTS],
            const int r_idx,
            const int a_idx)
        {
            static_assert(LA::OUTPUTS == OUTPUTS_A);
            static_assert(LB::OUTPUTS == OUTPUTS_B);
            static_assert(LA::OUTPUTS % Vec16s::size() == 0);
            static_assert(LB::OUTPUTS % Vec16s::size() == 0);

            int update_layer_b = 0;
            for (int i = 0; i < r_idx; ++i)
                update_layer_b += remove_inputs[i] < LB::INPUTS;
            for (int i = 0; i < a_idx; ++i)
                update_layer_b += add_inputs[i] < LB::INPUTS;

            Vec16s vo, vw;

            /* layer A */
            for (int j = 0; j != OUTPUTS_A; j += Vec16s::size())
            {
                vo.load_a(&_output_a[j]);

                for (int i = 0; i < r_idx; ++i)
                {
                    const auto index = remove_inputs[i];
                    ASSERT(index < LA::INPUTS);
                    vw.load_a(&layer_a._w[index][j]);
                    vo -= vw;
                }

                for (int i = 0; i < a_idx; ++i)
                {
                    const auto index = add_inputs[i];
                    ASSERT(index < LA::INPUTS);
                    vw.load_a(&layer_a._w[index][j]);
                    vo += vw;
                }
                vo.store_a(&_output_a[j]);
            }

            if (update_layer_b)
            {
                /* layer B */
                for (int j = 0; j != OUTPUTS_B; j += Vec16s::size())
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

        /** Incremental update of one-hot encoding */
        INLINE void update(
            const State& from_pos,
            const State& to_pos,
            const Move& move,
            Color color, /* color of side that moved */
            int (&remove)[INPUTS],
            int (&add)[INPUTS],
            int& r_idx,
            int& a_idx)
        {
            if (to_pos.promotion)
            {
                // add the promoted-to piece
                ASSERT(move.promotion() == to_pos.promotion);
                delta(add, a_idx, to_pos.promotion, color, move.to_square());

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

            if (to_pos.capture_value)
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
        constexpr size_t POOL_STRIDE = 4;

        static_assert(A::OUTPUTS_A == L2::INPUTS * POOL_STRIDE);
        static_assert(A::OUTPUTS_B == ATTN::INPUTS);

        ALIGN float attn_in[ATTN::INPUTS];
        ALIGN float attn_out[ATTN::OUTPUTS];
        ALIGN float l1_out[A::OUTPUTS_A];
        ALIGN float l2_in[L2::INPUTS];
        ALIGN float l2_out[L2::OUTPUTS];
        ALIGN float l3_out[L3::OUTPUTS];
        ALIGN float output[1];

        activation(a._output_a, l1_out); // process output of hidden_1a
        activation(a._output_b, attn_in); // process output of hidden_1b

        pool(l1_out, l2_in);

        /*
         * The dynamic weights computed by the "attention" layer
         * are used to modulate the output of another hidden layer
         * through element-wise multiplication.
         */
        attn.dot(attn_in, attn_out);

        static_assert(L2::INPUTS % Vector::size() == 0);

    #if true /* vectorized */
        Vector v1, v2;
        for (int i = 0; i != L2::INPUTS; i += Vector::size())
        {
            v1.load_a(&l2_in[i]);
            v2.load_a(&attn_out[i % ATTN::OUTPUTS]);
            (v1 * v2).store_a(&l2_in[i]);
        }
    #else
        for (int i = 0; i < A::OUTPUTS_A; ++i)
        {
            l2_in[i] *= attn_out[i % ATTN::OUTPUTS];
        }
    #endif /* !vectorized */

        static const Vector v_zero(0.0);

        l2.dot(l2_in, l2_out, [](const Vector& v) { return max(v, v_zero); });
        l3.dot(l2_out, l3_out, [](const Vector& v) { return max(v, v_zero); });

        out.dot(l3_out, output);
        return 100 * output[0];
    }


    /**
     * Evaluate FEN from White's point of view, for testing.
     */
    int eval_fen(const std::string&);

#if !WITH_NNUE
    INLINE int eval_fen(const std::string&) { return 0; }
#endif

} /* namespace nnue */
