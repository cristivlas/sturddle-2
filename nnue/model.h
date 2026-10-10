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
#include "nnue.h"

namespace nnue
{
    /* Define the network architecture
     *
     *  768 piece-square inputs (white view)   same inputs, black view (idx ^ 120)
     *           |  bucket b, rows of L1A                      |  bucket mirror(b)
     *           v                                             v
     *    white half: 1024 x int16                      black half: 1024 x int16
     *           \_____________________________________________/
     *                                   |
     *           accumulator [black 1024 | white 1024 ]   int16 at 1/nnue::QSCALE
     *                                   |  clamp 0..1023, >> 3
     *                                   v
     *                         activations  2048 x u8  (1.0 == 128)
     *                                   |  stack[side to move]
     *                                   v
     *              L2  2048 -> 32  s8 weights, int32 sums   ->  relu -> float
     *              L3    32 -> 32  float                    ->  relu
     *              EVAL  32 -> 1   float                    ->  centipawns x 100
     */
    constexpr int INPUTS_A = nnue::ACTIVE_INPUTS * nnue::NUM_BUCKETS;
    constexpr int HIDDEN_1A = 1024; /* per perspective */
    constexpr int HIDDEN_2 = 32;
    constexpr int HIDDEN_3 = 32;
    constexpr int STACKS = 2; /* selected by side to move, black first */
    constexpr int L2_SCALE = 128; /* hidden_2 s8 weights */

    using L1AType = nnue::Layer<INPUTS_A, HIDDEN_1A, int16_t, nnue::QSCALE, true /* incremental */>;
    using L2Type = nnue::Layer<2 * HIDDEN_1A, HIDDEN_2, int8_t, L2_SCALE, false, nnue::ACT_SCALE>;
#if USE_BF16 && NNUE_TAIL_BF16
    using L3Type = nnue::Layer<HIDDEN_2, HIDDEN_3, __bf16>;
#else
    using L3Type = nnue::Layer<HIDDEN_2, HIDDEN_3>;
#endif /* USE_BF16 && NNUE_TAIL_BF16 */
    using EVALType = nnue::Layer<HIDDEN_3, 1>;

    struct Model
    {
       /*
        * The accumulator holds both perspectives of L1A, [black view, white view];
        * the side to move selects one of the L2 -> L3 -> EVAL stacks.
        */
        using Accumulator = nnue::Accumulator<INPUTS_A, HIDDEN_1A>;

        static constexpr size_t param_count()
        {
            return L1AType::param_count()
                + STACKS * (L2Type::param_count() + L3Type::param_count() + EVALType::param_count());
        }

        void init();

        void load_weights(const std::filesystem::path& weights_path);

        std::string default_weights_path;

        template <typename Ctxt>
        INLINE void update(Accumulator& accumulator, const Ctxt* ctxt)
        {
            accumulator.update(L1A, ctxt->state());
        }

        /* Incremental */
        template <typename Ctxt>
        INLINE void update(Accumulator& accumulator, const Ctxt* ctxt, Accumulator& prev_acc, Accumulator::RefreshTable& refresh)
        {
            accumulator.update(L1A, ctxt->_parent->state(), ctxt->state(), ctxt->_move, prev_acc, refresh);
        }

        INLINE int eval(const Accumulator& acc, bool stm) const
        {
            return ::nnue::eval(acc, L2[stm], L3[stm], EVAL[stm]);
        }

    private:
        void validate_weights_file(const std::filesystem::path& weights_path);
        void load_layers(std::istream& file);

        L1AType L1A;
        L2Type L2[STACKS];
        L3Type L3[STACKS];
        EVALType EVAL[STACKS];
    };
}
