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

#define ALIGN alignas(32)

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

        int8_t _input[INPUTS] = { 0 }; /* one-hot encoding */
        ALIGN float _output[OUTPUTS] = { 0 };

        template <typename L> INLINE void add(const L& layer, int i)
        {
            static_assert(L::OUTPUTS == OUTPUTS);
            #pragma clang loop vectorize(enable)
            for (int j = 0; j != layer.OUTPUTS; ++j)
                _output[j] += layer._wt[j][i];
        }

        template<typename L> INLINE void remove(const L& layer, int i)
        {
            static_assert(L::OUTPUTS == OUTPUTS);
            #pragma clang loop vectorize(enable)
            for (int j = 0; j != layer.OUTPUTS; ++j)
                _output[j] -= layer._wt[j][i];
        }

        template<typename L> INLINE void update(const L& layer, const State& state)
        {
            memset(&_input, 0, sizeof(_input));
            one_hot_encode(state, _input);
            layer.dot_product(_input, _output);
        }

        template<typename L> INLINE
        void update(const L& layer, const State& state, const Accumulator& prev)
        {
            memset(&_input, 0, sizeof(_input));
            one_hot_encode(state, _input);

            memcpy(_output, prev._output, sizeof(_output));
            for (int i = 0; i != INPUTS; ++i)
            {
                if (!_input[i] && prev._input[i])
                    remove(layer, i);
                if (_input[i] && !prev._input[i])
                    add(layer, i);
            }
        }
    };


    template <typename L> INLINE int eval(const Accumulator& a, const L& layer)
    {
        ALIGN float input[L::INPUTS];
        ALIGN float output[1];

        static_assert(L::INPUTS == Accumulator::OUTPUTS);
        #pragma clang loop vectorize(enable)
        for (int i = 0; i != Accumulator::OUTPUTS; ++i)
            input[i] = a._output[i];

        activation(input);

        layer.dot_product(input, output);
        return 100 * output[0];
    }

    int eval_fen(const std::string&);

} /* namespace nnue */

#undef ALIGN
