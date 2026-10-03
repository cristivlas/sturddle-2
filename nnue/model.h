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
    /* Define the network architecture */
    constexpr int INPUTS_A = nnue::ACTIVE_INPUTS * nnue::NUM_BUCKETS;
    constexpr int INPUTS_B = 256;
    constexpr int HIDDEN_1A = 2048;
    constexpr int HIDDEN_1A_POOLED = HIDDEN_1A / nnue::POOL_STRIDE;
    constexpr int HIDDEN_1B = HIDDEN_1A_POOLED; /* 1b modulates pooled 1:1 */
    constexpr int HIDDEN_2 = 16;
    constexpr int HIDDEN_3 = 16;

    using L1AType = nnue::Layer<INPUTS_A, HIDDEN_1A, int16_t, nnue::QSCALE, true /* incremental */>;

#if NNUE_HIDDEN_1C
    /* L1B and L1C are load-only; the accumulator uses their fusion L1M as its layer B */
    using L1BType = nnue::Layer<INPUTS_B, HIDDEN_1B, int16_t, nnue::QSCALE>;
    using L1CType = nnue::Layer<nnue::INPUTS_C, HIDDEN_1B, int16_t, nnue::QSCALE>;
    using L1MType = nnue::Layer<nnue::TURN_INDEX, HIDDEN_1B, int16_t, nnue::QSCALE, true /* incremental */>;
#else
    using L1BType = nnue::Layer<INPUTS_B, HIDDEN_1B, int16_t, nnue::QSCALE, true /* incremental */>;
#endif /* NNUE_HIDDEN_1C */

    using PoolType = nnue::PoolLayer<HIDDEN_1A>;

#if NNUE_L2_INT16
    using L2Type = nnue::Layer<HIDDEN_1A_POOLED, HIDDEN_2, int16_t, nnue::WQSCALE>;
#elif USE_BF16
    using L2Type = nnue::Layer<HIDDEN_1A_POOLED, HIDDEN_2, __bf16>;
#else
    using L2Type = nnue::Layer<HIDDEN_1A_POOLED, HIDDEN_2, float>;
#endif
    using L3Type = nnue::Layer<HIDDEN_2, HIDDEN_3>;
    using EVALType = nnue::Layer<HIDDEN_3, 1>;

    /* Move-prediction head (experimental): own 256-wide sub-accumulator off raw inputs
     * (decoupled from eval), then a bilinear map to the 4096 (from,to) logits scored
     * per-move by column. Full recompute per node -- only used at early iterations.
     */
    constexpr int MOVE_ACC = 256;

    using LMOVEAccType = nnue::Layer<INPUTS_A / nnue::NUM_BUCKETS, MOVE_ACC, int16_t, nnue::QSCALE>;
    using LMOVEType = nnue::Layer<MOVE_ACC, 4096, int16_t, nnue::QSCALE>;

    struct Model
    {
       /*
        * The accumulator takes the inputs and processes them into two outputs,
        * using layers L1A and L1B. L1B processes the 1st 256 inputs, which
        * correspond to kings and pawns. The (linear) output of L1B modulates the
        * pooled output of L1A 1:1. With NNUE_HIDDEN_1C, L1C (bishops and occupancy)
        * is also linear; L1B and L1C are fused at load into L1M, which maps every
        * piece-square input to one row.
        */
        using Accumulator = nnue::Accumulator<INPUTS_A, HIDDEN_1A, HIDDEN_1B>;

        void init();

        void validate_weights_file(const std::filesystem::path& weights_path);
        void load_weights(const std::filesystem::path& weights_path);

    #if NNUE_HIDDEN_1C
        INLINE void fuse_modulation()
        {
            ::nnue::check_modulation_bounds(L1B, L1C);
            ::nnue::fuse_modulation(L1M, L1B, L1C);
        }
    #endif /* NNUE_HIDDEN_1C */

        std::string default_weights_path;

        template <typename Ctxt>
        INLINE void update(Accumulator& accumulator, const Ctxt* ctxt)
        {
        #if NNUE_HIDDEN_1C
            accumulator.update(L1A, L1M, ctxt->state());
            check_fused_modulation(accumulator, ctxt->state());
        #else
            accumulator.update(L1A, L1B, ctxt->state());
        #endif /* NNUE_HIDDEN_1C */
        }


        /* Incremental */
        template <typename Ctxt>
        INLINE void update(Accumulator& accumulator, const Ctxt* ctxt, Accumulator& prev_acc, Accumulator::RefreshTable& refresh)
        {
        #if NNUE_HIDDEN_1C
            accumulator.update(L1A, L1M, ctxt->_parent->state(), ctxt->state(), ctxt->_move, prev_acc, refresh);
            check_fused_modulation(accumulator, ctxt->state());
        #else
            accumulator.update(L1A, L1B, ctxt->_parent->state(), ctxt->state(), ctxt->_move, prev_acc, refresh);
        #endif /* NNUE_HIDDEN_1C */
        }


        /* The fused L1M output must equal L1B + L1C computed separately */
        INLINE void check_fused_modulation(const Accumulator& accumulator, const chess::State& state)
        {
        #if NNUE_HIDDEN_1C && DEBUG_INCREMENTAL
            ALIGN input_t input_b[round_up<INPUT_STRIDE>(ACTIVE_INPUTS)] = { };
            ALIGN input_t input_c[INPUTS_C] = { };
            one_hot_encode(state, input_b);
            encode_bishops_occupancy(state, input_c);

            ALIGN int16_t output_b[HIDDEN_1B], output_c[HIDDEN_1B];
            L1B.dot(input_b, output_b);
            L1C.dot(input_c, output_c);

            for (int j = 0; j != HIDDEN_1B; ++j)
                ASSERT_ALWAYS(int16_t(output_b[j] + output_c[j]) == accumulator._output_b[j]);
        #endif /* NNUE_HIDDEN_1C && DEBUG_INCREMENTAL */
        }

        INLINE int eval(const Accumulator& acc, bool stm) const
        {
            return ::nnue::eval(acc, POOL, L2, L3, EVAL, stm);
        }

        L1AType L1A;
        L1BType L1B;
    #if NNUE_HIDDEN_1C
        L1CType L1C;
        L1MType L1M;
    #endif /* NNUE_HIDDEN_1C */
        PoolType POOL;
        L2Type L2;
        L3Type L3;
        EVALType EVAL;

    #if USE_MOVE_PREDICTION
        LMOVEAccType LMOVE_ACC;
        LMOVEType LMOVES;
    #endif /* USE_MOVE_PREDICTION */
    };
}
