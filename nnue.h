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
#include <immintrin.h>

#define ALIGN alignas(32)

#if __AVX2__
/*
 * https://stackoverflow.com/questions/23189488/
 * horizontal-sum-of-32-bit-floats-in-256-bit-avx-vector
 */
static INLINE float _mm256_reduce_add_ps(__m256 x)
{
    // ( x3+x7, x2+x6, x1+x5, x0+x4 )
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    // ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 )
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    // ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 )
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}
#endif /* __AVX2__ */


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

    template <typename U, typename V, int N>
    INLINE void activation(const U (&input)[N], V (&output)[N])
    {
        #pragma clang loop vectorize(enable)
        for (int i = 0; i != N; ++i)
            output[i] = clipped_relu(input[i]);
    }

    template <int N, int M, typename T=float, int Scale=1>
    struct Layer
    {
        static constexpr int INPUTS = N;
        static constexpr int OUTPUTS = M;
        static constexpr float scale = Scale;

        ALIGN T _b[OUTPUTS]; /* biases */
        ALIGN T _w[INPUTS][OUTPUTS]; /* weights */
        ALIGN T _wt[OUTPUTS][INPUTS]; /* weights transposed */

        Layer(const float(&w)[INPUTS][OUTPUTS], const float(&b)[OUTPUTS])
        {
            for (int j = 0; j != OUTPUTS; ++j)
                _b[j] = b[j] * Scale;

            for (int i = 0; i != INPUTS; ++i)
                for (int j = 0; j != OUTPUTS; ++j)
                    _w[i][j] = _wt[j][i] = w[i][j] * Scale;
        }

        /* input */
        static INLINE void dot(
            const int8_t(&input)[INPUTS],
            float(&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS]
        )
        {
            static_assert(Scale == 1);

        #if __AVX2__
            static_assert(INPUTS % 8 == 0);
            for (int j = 0; j != OUTPUTS; ++j)
            {
                output[j] = b[j];
                __m256 sum = _mm256_setzero_ps();
                for (int i = 0; i < INPUTS; i += 8)
                {
                    // load the first 4 8-bit integers into a __m128i
                    __m128i packed_8_bit_ints1 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&input[i]));

                    // extend the first 4 8-bit integers to 32-bit integers
                    __m128i extended_ints1 = _mm_cvtepu8_epi32(packed_8_bit_ints1);

                    // convert the first 4 32-bit integers to floating-point values
                    __m128 floats1 = _mm_cvtepi32_ps(extended_ints1);

                    // load the second 4 8-bit integers into a __m128i
                    __m128i packed_8_bit_ints2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>((&input[i + 4])));

                    // extend the second 4 8-bit integers to 32-bit integers
                    __m128i extended_ints2 = _mm_cvtepu8_epi32(packed_8_bit_ints2);

                    // convert the second 4 32-bit integers to floating-point values
                    __m128 floats2 = _mm_cvtepi32_ps(extended_ints2);

                    // pack the two __m128 values into a __m256
                    __m256 va = _mm256_set_m128(floats2, floats1);

                    // load transposed weights
                    __m256 vb = _mm256_load_ps(&wt[j][i]);

                    sum = _mm256_fmadd_ps(va, vb, sum);
                }
                output[j] += _mm256_reduce_add_ps(sum);
            }
        #else
            for (int j = 0; j != OUTPUTS; ++j)
            {
                output[j] = b[j];
                for (int i = 0; i != INPUTS; ++i)
                    output[j] += input[i] * wt[j][i];
            }
        #endif /* __AVX2__ */
        }

        /* hidden */
        static INLINE void dot(
            const float(&input)[INPUTS],
            float(&output)[OUTPUTS],
            const float(&b)[OUTPUTS],
            const float(&wt)[OUTPUTS][INPUTS]
        )
        {
            using Vector = Vec16f;
            static_assert(INPUTS % Vector::size() == 0);
            static_assert(OUTPUTS % Vector::size() == 0);
            static_assert(Scale == 1);

            Vector v_in, v_out, v_wt;

            #pragma clang loop vectorize(enable)
            for (int j = 0; j < OUTPUTS; j += Vector::size())
            {
                v_out.load(&b[j]);

                #pragma clang loop vectorize(enable)
                for (int i = 0; i < INPUTS; i += Vector::size())
                {
                    v_in.load(&input[i]);
                    v_wt.load(&wt[j][i]);
                    v_out += v_in * v_wt;
                }
                v_out.store(&output[j]);
            }
        }

        /* output */
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

        #if __AVX2__
                Vec16s sum(0);
                static_assert(INPUTS % 16 == 0);
                for (int i = 0; i != INPUTS; i += 16)
                {
                    const auto v0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(&input[i]), vs));
                    const auto v1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(&input[i + 8]), vs));
                    /*
                     * https://stackoverflow.com/questions/20918987/
                     * why-is-permute-needed-in-parallel-simd-sse-avx
                     */
                    const auto va = _mm256_permutevar8x32_epi32(
                        _mm256_packs_epi32(v0, v1),
                        _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0)
                    );
                    const auto vb = _mm256_load_si256(reinterpret_cast<const __m256i*>(&wt[j][i]));
                    sum = _mm256_adds_epi16(sum, _mm256_mullo_epi16(va, vb));
                }
                output[j] += horizontal_add(extend_low(sum));
                output[j] += horizontal_add(extend_high(sum));
        #else
                #pragma clang loop vectorize(enable)

                for (int i = 0; i != INPUTS; ++i)
                    output[j] += scale * input[i] * wt[j][i];

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


    template <int N, int M> struct Accumulator
    {
        static constexpr int INPUTS = N;
        static constexpr int OUTPUTS = M;

        ALIGN int8_t _input[INPUTS] = { 0 }; /* one-hot encoding */
        ALIGN float _output[OUTPUTS] = { 0 };
        uint64_t _hash = 0;
        uint64_t _clock = 0;

        template <typename L> INLINE void add(const L& layer, int i)
        {
            static_assert(L::OUTPUTS == OUTPUTS);

            using Vector = Vec16f;
            static_assert(OUTPUTS % Vector::size() == 0);

            Vector vo, vw;
            #pragma clang loop vectorize(enable)
            for (int j = 0; j != layer.OUTPUTS; j += Vector::size())
            {
                vo.load(&_output[j]);
                vw.load(&layer._w[i][j]);
                vo += vw;
                vo.store(&_output[j]);
            }
        }

        template<typename L> INLINE void remove(const L& layer, int i)
        {
            static_assert(L::OUTPUTS == OUTPUTS);

            using Vector = Vec16f;
            static_assert(OUTPUTS % Vector::size() == 0);

            Vector vo, vw;
            #pragma clang loop vectorize(enable)
            for (int j = 0; j != layer.OUTPUTS; j += Vector::size())
            {
                vo.load(&_output[j]);
                vw.load(&layer._w[i][j]);
                vo -= vw;
                vo.store(&_output[j]);
            }
        }

        /** Compute 1st layer output from scratch at root */
        template <typename L> INLINE void update(const L& layer, const State& state)
        {
            if (state.hash() != _hash)
            {
                _hash = state.hash();
                memset(&_input, 0, sizeof(_input));
                one_hot_encode(state, _input);

                layer.dot(_input, _output);

                ++_clock;
            }
        }

        /** Update 1st layer output incrementally, based on a previous state */
        template <typename L, typename A>
        INLINE void update(const L& layer, const State& state, const A& ancestor, int8_t(&temp)[INPUTS])
        {
            if (state.hash() != _hash)
            {
                _hash = state.hash();
                one_hot_encode(state, temp);

                ASSERT(ancestor._clock);

                /* part of the same search? use own previous state */
                if (_clock == ancestor._clock)
                {
                    for (int i = 0; i != INPUTS; ++i)
                    {
                        if (!temp[i] && _input[i]) /* 1 -> 0 */
                            remove(layer, i);
                        if (temp[i] && !_input[i]) /* 0 -> 1*/
                            add(layer, i);
                        _input[i] = temp[i];
                    }
                }
                else
                {
                    /* compute delta based on ancestor state */
                    _clock = ancestor._clock;
                    memcpy(_output, ancestor._output, sizeof(_output));

                    for (int i = 0; i != INPUTS; ++i)
                    {
                        if (!temp[i] && ancestor._input[i])
                            remove(layer, i);
                        if (temp[i] && !ancestor._input[i])
                            add(layer, i);
                        _input[i] = temp[i];
                    }
                }
            }
        }
    };


    template <typename A, typename L> INLINE int eval(const A& a, const L& layer)
    {
        ALIGN float input[L::INPUTS];
        ALIGN float output[1];

        static_assert(L::INPUTS == A::OUTPUTS);
        static_assert(sizeof(input) == sizeof(a._output));

        activation(a._output, input);

        layer.dot(input, output);
        return 100 * output[0];
    }

    int eval_fen(const std::string&);

} /* namespace nnue */

#undef ALIGN
