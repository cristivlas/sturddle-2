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

namespace nnue
{
    using namespace chess;

    constexpr bool debug_incremental = false;

#if INSTRSET >= 9
    using Vector = Vec16f;

    INLINE Vector load_vec(const int8_t* input)
    {
    #if 0
        return Vector(
            input[0], input[1], input[2], input[3],
            input[4], input[5], input[6], input[7],
            input[8], input[9], input[10],input[11],
            input[12],input[13],input[14],input[15]);
    #else
        Vec16c v;
        v.load_a(input);
        return to_float(extend(extend(v)));
    #endif
    }
#elif INSTRSET >= 8
    using Vector = Vec8f;

    INLINE Vector load_vec(const int8_t* input)
    {
    #if 1
        return Vector(
            input[0], input[1], input[2], input[3],
            input[4], input[5], input[6], input[7]);
    #else
        __m128i input_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));

        // Sign-extend the 8-bit integers to 32-bit integers
        __m256i input_epi32 = _mm256_cvtepi8_epi32(input_vec);

        // Convert the 32-bit integers to floats
        return Vector(_mm256_cvtepi32_ps(input_epi32));
    #endif
    }
#else
    using Vector = Vec4f;

    INLINE Vector load_vec(const int8_t* input)
    {
        return Vector(input[0], input[1], input[2], input[3]);
    }
#endif /* INSTRSET */

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
    INLINE constexpr int psi(PieceType piece_type, Color color, Square square)
    {
        return (piece_type % 6) * 128 + (64 * color) + 63 - square;
    }


    /** Rectified Linear Unit (reLU) activation */
    static const Vector v_zero(0.0);

    template <int N>
    INLINE void activation(const float (&input)[N], float* output)
    {
        static_assert(N % Vector::size() == 0);

        Vector v;
        for (int i = 0; i != N; i += Vector::size())
        {
            v.load_a(&input[i]);
            max(v, v_zero).store_a(&output[i]);
        }
    }


    template <int I, int O, typename T=float>
    struct Layer
    {
        static constexpr int INPUTS = I;
        static constexpr int OUTPUTS = O;

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
        template <typename F>
        static INLINE void dot(
            const int8_t* input,
            float* output,
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS],
            F /* dummy activation */
        )
        {
        #if 0
            for (int j = 0; j != OUTPUTS; ++j)
            {
                output[j] = b[j];
                #pragma clang loop vectorize(enable)
                for (int i = 0; i != INPUTS; ++i)
                    output[j] += input[i] * wt[j][i];
            }
        #else /* vector */
            static_assert(OUTPUTS % Vector::size() == 0);

            constexpr auto N = Vector::size();
            constexpr auto R = round_down<Vector::size()>(INPUTS);

            for (int j = 0; j != OUTPUTS; j += N)
            {
                Vector vw, sum[N];

                for (int k = 0; k != N; ++k)
                    sum[k] = Vector(0.0);

                for (int i = 0; i != R; i += Vector::size())
                {
                    const Vector in(load_vec(input + i));

                    for (int k = 0; k != N; ++k)
                    {
                        vw.load(&wt[j + k][i]);
                        sum[k] = mul_add(in, vw, sum[k]);
                    }
                }

                for (int k = 0; k != N; ++k)
                {
                    float r = 0;
                    for (int i = R; i != INPUTS; ++i)
                        r += input[i] * wt[j + k][i];
                    output[j + k] = b[j + k] + r + horizontal_add(sum[k]);
                }
            }
        #endif /* vector */
        }

        /* output */
        template<typename F>
        static INLINE void dot(
            const float* input,
            float* output,
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS],
            F activate
        )
        {
            static_assert(INPUTS % Vector::size() == 0);

            if constexpr(OUTPUTS % Vector::size() == 0)
            {
                constexpr int N = Vector::size();

                for (int j = 0; j != OUTPUTS; j += N)
                {
                    Vector sum[N], out, v_wt, v_in;

                    for (int k = 0; k != N; ++k)
                        sum[k] = Vector(0.0);

                    for (int i = 0; i != INPUTS; i += Vector::size())
                    {
                        v_in.load_a(&input[i]);
                        for (int k = 0; k != N; ++k)
                        {
                            v_wt.load_a(&wt[j + k][i]);
                            sum[k] = mul_add(v_in, v_wt, sum[k]);
                        }
                    }
                #if 0
                    for (int k = 0; k != N; ++k)
                        out.insert(k, activate(b[j + k] + horizontal_add(sum[k])));

                    out.store_a(&output[j]);
                #else
                    for (int k = 0; k != N; ++k)
                        output[j + k] = activate(b[j + k] + horizontal_add(sum[k]));
                #endif
                }
            }
            else
            {
                for (int j = 0; j != OUTPUTS; ++j)
                {
                    Vector sum(0.0);
                    Vector v_in, v_wt;

                    for (int i = 0; i != INPUTS; i += Vector::size())
                    {
                        v_in.load_a(&input[i]);
                        v_wt.load_a(&wt[j][i]);
                        sum = mul_add(v_in, v_wt, sum);
                    }
                    output[j] = activate(b[j] + horizontal_add(sum));
                }
            }
        }

        template <typename U, typename V>
        INLINE void dot(const U* input, V* output) const
        {
            dot(input, output, _b, _wt, [](V v) { return v; });
        }

        template <typename U, typename V, typename F>
        INLINE void dot(const U* input, V* output, F activate) const
        {
            dot(input, output, _b, _wt, activate);
        }
    };


    template <int N, int M, int O> struct Accumulator
    {
        static constexpr int INPUTS = N;
        static constexpr int OUTPUTS_A = M;
        static constexpr int OUTPUTS_B = O;

        /* bit index of the side-to-move feature within one-hot encoding */
        static constexpr int TURN_INDEX = INPUTS - 1;

        int8_t _input[INPUTS] = { 0 }; /* one-hot encoding */
        ALIGN float _output_a[OUTPUTS_A] = { 0 };
        ALIGN float _output_b[OUTPUTS_B] = { 0 };
        uint64_t _hash = 0;

        /** Compute 1st layer output from scratch at root */
        template <typename LA, typename LB>
        INLINE void update(const LA& layer_1a, const LB& layer_1b, const State& state)
        {
            if (state.hash() != _hash)
            {
                _hash = state.hash();

                memset(&_input, 0, sizeof(_input));
                one_hot_encode(state, _input);

                layer_1a.dot(_input, _output_a);
                layer_1b.dot(_input, _output_b);
            }
        }

        /** Update 1st layer output incrementally, based on a previous state */
        template <typename LA, typename LB, typename A>
        INLINE void update(
            const LA& layer_a,
            const LB& layer_b,
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

                memcpy(_output_a, ancestor._output_a, sizeof(_output_a));
                memcpy(_output_b, ancestor._output_b, sizeof(_output_b));
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

                if constexpr(debug_incremental)
                {
                    int8_t temp[INPUTS] = { 0 };
                    one_hot_encode(state, temp);

                    for (int i = 0; i != INPUTS; ++i)
                        ASSERT_ALWAYS(_input[i] == temp[i]);
                }

                if (state.turn)
                    add_inputs[a_idx++] = TURN_INDEX;
                else
                    remove_inputs[r_idx++] = TURN_INDEX;

                recompute(layer_a, layer_b, remove_inputs, add_inputs, r_idx, a_idx);

                if constexpr(debug_incremental)
                {
                    float output_a[OUTPUTS_A] = { 0 };
                    layer_a.dot(_input, output_a);

                    for (int i = 0; i != OUTPUTS_A; ++i)
                        ASSERT_ALWAYS(abs(output_a[i] - _output_a[i]) < 0.0001);

                    float output_b[OUTPUTS_B] = { 0 };
                    layer_b.dot(_input, output_b);
                    for (int i = 0; i != OUTPUTS_B; ++i)
                        ASSERT_ALWAYS(abs(output_b[i] - _output_b[i]) < 0.0001);
                }
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
            static_assert(LA::OUTPUTS % Vec16f::size() == 0);
            static_assert(LB::OUTPUTS % Vec16f::size() == 0);

            Vec16f vo, vw;
            bool update_layer_b = false;
            /* layer A */
            for (int j = 0; j != OUTPUTS_A; j += Vec16f::size())
            {
                vo.load_a(&_output_a[j]);

                for (int i = 0; i < r_idx; ++i)
                {
                    const auto index = remove_inputs[i];
                    update_layer_b |= index < LB::INPUTS;
                    ASSERT(index < LA::INPUTS);
                    vw.load_a(&layer_a._w[index][j]);
                    vo -= vw;
                }

                for (int i = 0; i < a_idx; ++i)
                {
                    const auto index = add_inputs[i];
                    update_layer_b |= index < LB::INPUTS;
                    ASSERT(index < LA::INPUTS);
                    vw.load_a(&layer_a._w[index][j]);
                    vo += vw;
                }
                vo.store_a(&_output_a[j]);
            }

            if (update_layer_b)
            {
                /* layer B */
                for (int j = 0; j != OUTPUTS_B; j += Vec16f::size())
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
                            continue;
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

    template <typename A, typename L2, typename L3, typename L4>
    INLINE int eval(const A& a, const L2& l2, const L3& l3, const L4& l4)
    {
        ALIGN float output1[L2::INPUTS];
        ALIGN float output2[L3::INPUTS];
        ALIGN float output3[L4::INPUTS];
        ALIGN float output4[1];

        static_assert(L2::INPUTS == A::OUTPUTS_A);

        activation(a._output_a, output1);
        l2.dot(output1, output2, [](float v){ return std::max<float>(v, 0); });

        activation(a._output_b, &output2[L2::OUTPUTS]);

        l3.dot(output2, output3, [](float v){ return std::max<float>(v, 0); });
        l4.dot(output3, output4);

        return 100 * output4[0];
    }


    int eval_fen(const std::string&);

} /* namespace nnue */
