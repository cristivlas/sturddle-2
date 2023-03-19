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
#include "vectorclass.h"

#define ALIGN alignas(64)
#define DEBUG_INCREMENTAL false

namespace nnue
{
    using namespace chess;
    using Vector = Vec16f;

#if INSTRSET >= 10  // AVX512
    constexpr int unroll_factor = 8;
#elif INSTRSET >= 8  // AVX2
    constexpr int unroll_factor = 4;
#elif INSTRSET >= 4  // SSE3
    constexpr int unroll_factor = 2;
#else
    constexpr int unroll_factor = 1;
#endif

    template <typename V, std::size_t N>
    INLINE V sum_vectors(const V (&input)[N])
    {
        static_assert(N == unroll_factor);

        V sum;
        if constexpr (unroll_factor == 2)
        {
            sum = input[0] + input[1];
        }
        else if constexpr (unroll_factor == 4)
        {
            V temp1 = input[0] + input[1];
            V temp2 = input[2] + input[3];
            sum = temp1 + temp2;
        }
        else if constexpr (unroll_factor == 8)
        {
            V temp1 = input[0] + input[1];
            V temp2 = input[2] + input[3];
            V temp3 = input[4] + input[5];
            V temp4 = input[6] + input[7];
            V temp5 = temp1 + temp2;
            V temp6 = temp3 + temp4;
            sum = temp5 + temp6;
        }
        else
            return input[0];

        return sum;
    }


    template<unsigned int N>
    constexpr unsigned int round_down(unsigned int x)
    {
        return (x / N) * N;
    }

    template <typename T>
    INLINE void one_hot_encode(const State& board, T (&encoding)[769])
    {
        const auto color_masks = { board.occupied_co(BLACK), board.occupied_co(WHITE) };

        int i = 63;
        #pragma clang loop vectorize(enable)
        for (const auto bb : {
            board.kings, board.pawns, board.knights, board.bishops, board.rooks, board.queens })
        {
            for (const auto mask : color_masks)
            {
                for_each_square_r((bb & mask), [&](Square j) {
                    encoding[i - j] = 1;
                });
                i += 64;
            }
        }
        encoding[768] = board.turn;
    }

    /** Calculate the piece-square index into the one-hot encoding. */
    INLINE int psi(PieceType piece_type, Color color, Square square)
    {
        static constexpr int index[] = { 1, 2, 3, 4, 5, 0 };
        return index[piece_type - 1] * 128 + (64 * color) + 63 - square;
    }

    /** Rectified Linear Unit (reLU) activation */
#if 0
    template <typename U, typename V, int N>
    INLINE void activation(const U (&input)[N], V (&output)[N])
    {
        #pragma clang loop vectorize(enable)
        for (int i = 0; i != N; ++i)
            output[i] = std::max<V>(0, input[i]);
    }
#else
    template <int N>
    INLINE void activation(const float (&input)[N], float (&output)[N])
    {
        static_assert(N % Vector::size() == 0);
        static const Vector zero(0.0);

        Vector v;
        for (int i = 0; i != N; i += Vector::size())
        {
            v.load_a(&input[i]);
            max(v, zero).store_a(&output[i]);
        }
    }
#endif

    template <int N, int M, typename T=float>
    struct Layer
    {
        static constexpr int INPUTS = N;
        static constexpr int OUTPUTS = M;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _w[INPUTS][OUTPUTS]; /* weights */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */

        Layer(const float(&w)[INPUTS][OUTPUTS], const float(&b)[OUTPUTS])
        {
            for (int j = 0; j != OUTPUTS; ++j)
                _b[j] = b[j];

            for (int i = 0; i != INPUTS; ++i)
                for (int j = 0; j != OUTPUTS; ++j)
                    _w[i][j] = _wt[j][i] = w[i][j];
        }

        /* input */
        template <typename V>
        static INLINE void dot(
            const int8_t(&input)[INPUTS],
            V(&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS]
        )
        {
            for (int j = 0; j != OUTPUTS; ++j)
            {
            #if 0
                output[j] = b[j];
                #pragma clang loop vectorize(enable)
                for (int i = 0; i != INPUTS; ++i)
                    output[j] += input[i] * wt[j][i];
            #else
                Vector sum_unrolled[unroll_factor];
                for (int u = 0; u < unroll_factor; ++u)
                    sum_unrolled[u] = Vector(0.0);

                constexpr int R = round_down<Vector::size()>(INPUTS);

                for (int ii = 0; ii < R; ii += Vector::size() * unroll_factor)
                {
                    for (int i = ii; i < ii + Vector::size() * unroll_factor; i += Vector::size())
                    {
                        Vector vw;
                        vw.load(&wt[j][i]);
                        const Vector in(
                            input[i],   input[i+1], input[i+2], input[i+3],
                            input[i+4], input[i+5], input[i+6], input[i+7],
                            input[i+8], input[i+9], input[i+10],input[i+11],
                            input[i+12],input[i+13],input[i+14],input[i+15]);

                        int unroll_idx = (i - ii) / Vector::size();
                        sum_unrolled[unroll_idx] = mul_add(in, vw, sum_unrolled[unroll_idx]);
                    }
                }

                output[j] = b[j] + horizontal_add(sum_vectors(sum_unrolled));

                for (int i = R; i != INPUTS; ++i)
                    output[j] += input[i] * wt[j][i];

            #endif
            }
        }

        /* output */
        static INLINE void dot(
            const float(&input)[INPUTS],
            float(&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS]
        )
        {
            static_assert(INPUTS % (Vector::size() * unroll_factor) == 0,
                "Input size must be a multiple of vector size and unroll factor");

            for (int j = 0; j != OUTPUTS; ++j)
            {
                Vector sum(0.0);
                Vector v_in, v_wt;

                constexpr int unrolled_iterations = INPUTS / (Vector::size() * unroll_factor);

                for (int i = 0;
                    i < unrolled_iterations * unroll_factor * Vector::size();
                    i += unroll_factor * Vector::size())
                {
                    Vector sum_unrolled[unroll_factor];
                    for (int u = 0; u < unroll_factor; ++u)
                        sum_unrolled[u] = Vector(0.0);

                    for (int u = 0; u < unroll_factor; ++u)
                    {
                        v_in.load_a(&input[i + u * Vector::size()]);
                        v_wt.load_a(&wt[j][i + u * Vector::size()]);
                        sum_unrolled[u] = mul_add(v_in, v_wt, sum_unrolled[u]);
                    }

                    sum += sum_vectors(sum_unrolled);
                }
                output[j] = b[j] + horizontal_add(sum);
            }
        }

        template <typename U, typename V>
        INLINE void dot(const U(&input)[INPUTS], V(&output)[OUTPUTS]) const
        {
            dot(input, output, _b, _wt);
        }
    };


    template <int N, int M> struct Accumulator
    {
        static constexpr int INPUTS = N;
        static constexpr int OUTPUTS = M;

        /* bit index of the side-to-move feature within one-hot encoding */
        static constexpr int TURN_INDEX = INPUTS - 1;

        int8_t _input[INPUTS] = { 0 }; /* one-hot encoding */
        ALIGN float _output[OUTPUTS] = { 0 };
        uint64_t _hash = 0;

        /** Compute 1st layer output from scratch at root */
        template <typename L> INLINE void update(const L& layer, const State& state)
        {
            if (state.hash() != _hash)
            {
                _hash = state.hash();

                memset(&_input, 0, sizeof(_input));
                one_hot_encode(state, _input);

                layer.dot(_input, _output);
            }
        }

        /** Update 1st layer output incrementally, based on a previous state */
        template <typename L, typename A>
        INLINE void update(
            const L& layer,
            const State& prev,
            const State& state,
            const Move& move,
            const A& ancestor)
        {
            if (state.hash() != _hash)
            {
                _hash = state.hash();

                /* compute delta based on ancestor state */
                ASSERT(prev.turn != state.turn);

                memcpy(_output, ancestor._output, sizeof(_output));
                memcpy(_input, ancestor._input, sizeof(_input));

                int remove_inputs[INPUTS];
                int add_inputs[INPUTS];
                int r_idx = 0, a_idx = 0;

                if (move)
                {
                    update(prev, state, move, prev.turn, remove_inputs, add_inputs, r_idx, a_idx);
                    ASSERT(a_idx < INPUTS);
                    ASSERT(r_idx < INPUTS);

                    for (int i = 0; i != r_idx; ++i)
                        _input[remove_inputs[i]] = 0;
                    for (int i = 0; i != a_idx; ++i)
                        _input[add_inputs[i]] = 1;
                }
                _input[TURN_INDEX] ^= 1;

            #if DEBUG_INCREMENTAL
                int8_t temp[INPUTS] = { 0 };
                one_hot_encode(state, temp);

                for (int i = 0; i != INPUTS; ++i)
                    ASSERT_ALWAYS(_input[i] == temp[i]);
            #endif /* DEBUG_INCREMENTAL */

                if (state.turn)
                    add_inputs[a_idx++] = TURN_INDEX;
                else
                    remove_inputs[r_idx++] = TURN_INDEX;

                recalculate_output(layer, remove_inputs, add_inputs, r_idx, a_idx);

            #if DEBUG_INCREMENTAL
                float output[OUTPUTS] = { 0 };
                layer.dot(_input, output);
                for (int i = 0; i != OUTPUTS; ++i)
                {
                    std::cout << _output[i] << " " << output[i] << "\n";
                    ASSERT_ALWAYS(abs(output[i] - _output[i]) < 0.0001);
                }
            #endif /* DEBUG_INCREMENTAL */
            }
        }

        template <typename L>
        INLINE void recalculate_output(
            const L& layer,
            const int (&remove_inputs)[INPUTS],
            const int (&add_inputs)[INPUTS],
            const int r_idx,
            const int a_idx)
        {
            static_assert(L::OUTPUTS == OUTPUTS);
            static_assert(OUTPUTS % Vector::size() * unroll_factor == 0);

            constexpr int unrolled_iterations = (L::OUTPUTS / Vector::size()) / unroll_factor;

            Vector vo[unroll_factor], vw;

            for (int j = 0;
                j < unrolled_iterations * unroll_factor * Vector::size();
                j += unroll_factor * Vector::size())
            {
                for (int u = 0; u < unroll_factor; ++u)
                    vo[u].load_a(&_output[j + u * Vector::size()]);

                for (int i = 0; i < r_idx; ++i)
                {
                    for (int u = 0; u < unroll_factor; ++u)
                    {
                        vw.load_a(&layer._w[remove_inputs[i]][j + u * Vector::size()]);
                        vo[u] -= vw;
                    }
                }

                for (int i = 0; i < a_idx; ++i)
                {
                    for (int u = 0; u < unroll_factor; ++u)
                    {
                        vw.load_a(&layer._w[add_inputs[i]][j + u * Vector::size()]);
                        vo[u] += vw;
                    }
                }

                for (int u = 0; u < unroll_factor; ++u)
                    vo[u].store_a(&_output[j + u * Vector::size()]);
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
                add[a_idx++] = psi(to_pos.promotion, color, move.to_square());

                // remove the pawn
                remove[r_idx++] = psi(PieceType::PAWN, color, move.from_square());
            }
            else
            {
                const auto ptype = from_pos.piece_type_at(move.from_square());
                remove[r_idx++] = psi(ptype, color, move.from_square());
                add[a_idx++] = psi(ptype, color, move.to_square());

                if (to_pos.is_castle)
                {
                    const auto king_file = square_file(move.to_square());

                    remove[r_idx++] = psi(PieceType::ROOK, color, rook_castle_squares[king_file == 2][0][color]);
                    add[a_idx++] = psi(PieceType::ROOK, color, rook_castle_squares[king_file == 2][1][color]);
                }
            }

            if (to_pos.capture_value)
            {
                const auto capture_square = from_pos.is_en_passant(move)
                    ? Square(from_pos.en_passant_square - 8 * SIGN[color])
                    : move.to_square();
                const auto victim_type = from_pos.piece_type_at(capture_square);
                remove[r_idx++] = psi(victim_type, !color, capture_square);
            }
        }
    };


    template <typename A, typename L> INLINE int eval(const A& a, const L& layer)
    {
        ALIGN float input[L::INPUTS];
        ALIGN float output[1];

        static_assert(L::INPUTS == A::OUTPUTS);

        activation(a._output, input);

        layer.dot(input, output);
        return 100 * output[0];
    }


    int eval_fen(const std::string&);

} /* namespace nnue */
