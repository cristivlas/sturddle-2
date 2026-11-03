#pragma once
/* Piece grading evaluation helpers. */

#include "chess.h"

namespace
{
    using namespace chess;

#if EVAL_PIECE_GRADING
    static INLINE int eval_piece_grading_raw(const State& state)
    {
        int score = 0;

        const auto& adjust = ADJUST[pawn_bucket(state.pawns)];

        for (const auto color : { BLACK, WHITE })
        {
            const auto color_mask = state.occupied_co(color);

            score += SIGN[color] * (
                + popcount(state.pawns & color_mask) * adjust[PAWN]
                + popcount(state.knights & color_mask) * adjust[KNIGHT]
                + popcount(state.bishops & color_mask) * adjust[BISHOP]
                + popcount(state.rooks & color_mask) * adjust[ROOK]
                + popcount(state.queens & color_mask) * adjust[QUEEN]
            );
        }

        return score;
    }


    static INLINE int eval_piece_grading(const State& state)
    {
        if (state.grading_score == State::UNKNOWN_SCORE)
            state.grading_score = eval_piece_grading_raw(state);
        else
            ASSERT(state.grading_score == eval_piece_grading_raw(state));

        return state.grading_score;
    }
#endif /* EVAL_PIECE_GRADING */

} /* namespace */

