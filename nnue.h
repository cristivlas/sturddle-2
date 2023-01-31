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

        static INLINE void dot(
            const int8_t(&input)[INPUTS],
            float(&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS]
        )
        {
            static_assert(Scale == 1);

            for (int j = 0; j != OUTPUTS; ++j)
            {
                output[j] = b[j];
                for (int i = 0; i != INPUTS; ++i)
                    output[j] += input[i] * wt[j][i];
            }
        }

        static INLINE void dot(
            const float(&input)[INPUTS],
            float(&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS]
        )
        {
            static_assert(INPUTS % Vec4f::size() == 0);
            static_assert(OUTPUTS % Vec4f::size() == 0);
            static_assert(Scale == 1);

            Vec4f v_in, v_out, v_wt;
            for (int j = 0; j < OUTPUTS; j += Vec4f::size())
            {
                v_out.load(&b[j]);

                for (int i = 0; i < INPUTS; i += Vec4f::size())
                {
                    v_in.load(&input[i]);
                    v_wt.load(&wt[j][i]);
                    v_out += v_in * v_wt;
                }
                v_out.store(&output[j]);
            }
        }

       static INLINE void dot(
            const float(&input)[INPUTS],
            float(&output)[OUTPUTS],
            const int16_t(&b)[OUTPUTS],
            const int16_t(&wt)[OUTPUTS][INPUTS]
        )
        {
        #if __AVX2__
            const auto vs = _mm256_set1_ps(scale);
        #endif
            for (int j = 0; j != OUTPUTS; ++j)
            {
                output[j] = b[j] * scale;
            #if !__AVX2__
                for (int i = 0; i != INPUTS; ++i)
                    output[j] += scale * input[i] * wt[j][i];
            #else
                Vec16s sum(0);
                static_assert(INPUTS % 16 == 0);
                for (int i = 0; i != INPUTS; i += 16)
                {
                    const auto v0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(&input[i]), vs));
                    const auto v1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(&input[i + 8]), vs));
                    // https://stackoverflow.com/questions/20918987/why-is-permute-needed-in-parallel-simd-sse-avx
                    const auto va = _mm256_permutevar8x32_epi32(
                        _mm256_packs_epi32(v0, v1),
                        _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0)
                    );
                    const auto vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(&wt[j][i]));
                    sum = _mm256_adds_epi16(sum, _mm256_mullo_epi16(va, vb));
                }
                output[j] += horizontal_add(extend_low(sum));
                output[j] += horizontal_add(extend_high(sum));
            #endif /* __AVX2__ */
                output[j] /= scale * scale;
            }
        }

        template <typename V>
        INLINE void dot(const V(&input)[INPUTS], float(&output)[OUTPUTS]) const
        {
            dot(input, output, _b, _wt);
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
            layer.dot(_input, _output);
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

        layer.dot(input, output);
        return 100 * output[0];
    }

    int eval_fen(const std::string&);

} /* namespace nnue */

#undef ALIGN
