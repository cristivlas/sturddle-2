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

namespace nnue
{
    using namespace chess;

    template <typename T>
    INLINE void one_hot_encode(const State& state, T& encoding)
    {
        /* Iterate over the 64 squares on the chess board */
        for (int i = 0; i != 64; ++i)
        {
            int j = 0;
            if (const auto piece_type = state.piece_type_at(Square(i)))
            {
                const auto piece_color = state.piece_color_at(Square(i));
                j = piece_type + 6 * (piece_color != state.turn);
            }
            encoding[i * 13 + j] = 1;
        }
    }

    template <typename T> INLINE T clipped_relu(T x)
    {
        return std::min<T>(std::max<T>(0, x), 1.0);
    }

    template <typename V, int N> INLINE void activation(V (&output)[N])
    {
        for (int i = 0; i != N; ++i)
            output[i] = clipped_relu(output[i]);
    }

#define ALIGN alignas(32)

    template <int N, int M, typename T=float, int Scale=1>
    struct Layer
    {
        static constexpr int INPUTS = N;
        static constexpr int OUTPUTS = M;
        static constexpr float scale = Scale;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */

        Layer(const float(&w)[INPUTS][OUTPUTS], const float(&b)[OUTPUTS])
        {
            for (int j = 0; j != OUTPUTS; ++j)
                _b[j] = b[j] * Scale;

            for (int i = 0; i != INPUTS; ++i)
                for (int j = 0; j != OUTPUTS; ++j)
                    _wt[j][i] = w[i][j] * Scale;
        }

        template <typename V>
        INLINE void dot_product(const V(&input)[INPUTS], float(&output)[OUTPUTS]) const
        {
            for (int j = 0; j != OUTPUTS; ++j)
            {
                output[j] = _b[j] * scale;

                for (int i = 0; i != INPUTS; ++i)
                    output[j] += scale * input[i] * _wt[j][i];
                if constexpr(Scale > 1)
                    output[j] /= scale * scale;
            }
        }
    };

    struct Accumulator
    {
        static constexpr int INPUTS = 832;
        static constexpr int OUTPUTS = 256;

        int8_t _encoding[INPUTS] = { 0 };
        ALIGN int16_t output[OUTPUTS] = { 0 };
    };


    template <typename L1, typename L2>
    INLINE int eval(const Accumulator& a, const L1& l1, const L2& l2)
    {
        ALIGN float l2_input[L2::INPUTS];
        ALIGN float output[1];

        l1.dot_product(a._encoding, l2_input);
        activation(l2_input);

        l2.dot_product(l2_input, output);
        return 100 * output[0];
    }
#undef ALIGN

    int eval_fen(const std::string&);

} /* namespace nnue */
