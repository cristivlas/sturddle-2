#pragma once
/* Evaluation helpers called from search::Context::eval_tactical and other places. */

#include "chess.h"

namespace
{
    using namespace chess;
    using search::Context;


    /*
     * https://www.chessprogramming.org/images/7/70/LittleChessEvaluationCompendium.pdf
     * Grading of Pieces, page 4
     */
    static INLINE int eval_piece_grading(const State& state, int pcs)
    {
        double score = 0;
    #if 0
        const int p = popcount(state.pawns);

    #if !(TUNING_ENABLED || TUNING_PARTIAL)
        static constexpr
    #endif /* !(TUNING_ENABLED || TUNING_PARTIAL) */
        int piece_percents[4][4] = {
            { EVAL_KNIGHT_OPEN,     EVAL_BISHOP_OPEN,       EVAL_ROOK_OPEN,     EVAL_QUEEN_OPEN },
            { EVAL_KNIGHT_SEMIOPEN, EVAL_BISHOP_SEMIOPEN,   EVAL_ROOK_SEMIOPEN, EVAL_QUEEN_SEMIOPEN },
            { EVAL_KNIGHT_SEMICLOSE,EVAL_BISHOP_SEMICLOSE,  EVAL_ROOK_SEMICLOSE,EVAL_QUEEN_SEMICLOSE },
            { EVAL_KNIGHT_CLOSED,   EVAL_BISHOP_CLOSED,     EVAL_ROOK_CLOSED,   EVAL_QUEEN_CLOSED }
        };
        const auto& grading = piece_percents[int(p > 4) + int(p > 8) + int(p > 12)];
    #endif

        for (const auto color : { BLACK, WHITE })
        {
            const auto color_mask = state.occupied_co(color);

    #if 0
            score += SIGN[color] * (
                + popcount(state.knights & color_mask) * WEIGHT[KNIGHT] * grading[0]
                + popcount(state.bishops & color_mask) * WEIGHT[BISHOP] * grading[1]
                + popcount(state.rooks & color_mask) * WEIGHT[ROOK] * grading[2]
                + popcount(state.queens & color_mask) * WEIGHT[QUEEN] * grading[3]
            ) / 100.0;
    #endif
            score += SIGN[color] * popcount(state.pawns & color_mask) * interpolate(pcs, 0, ENDGAME_PAWN_BONUS);
        }

        return score;
    }

    /*----------------------------------------------------------------------
     * Tactical evaluations.
     * Hand-crafted evaluations are not compiled when using NNUE evals.
     * All tactical scores are computed from the white side's perspective.
     *----------------------------------------------------------------------*/

#if !WITH_NNUE /* HCE */
    static INLINE int eval_center(const State& state, int pc)
    {
        int attacks = 0;
        int occupancy = 0;

        for (auto color : { BLACK, WHITE })
        {
            const auto s = SIGN[color];
            occupancy += s * popcount(state.pawns & state.occupied_co(color) & BB_CENTER);

            for_each_square(BB_CENTER, [&](Square square) {
                attacks += s * popcount(state.pawns & BB_PAWN_ATTACKS[!color][square]);
            });

        }
        return attacks * interpolate(pc, CENTER_ATTACKS, 0)
            + occupancy * interpolate(pc, CENTER_OCCUPANCY, 0);
    }


    static INLINE Bitboard king_area(const State& state, Square king)
    {
        const auto f = square_file(king);
        const auto r = square_rank(king);

        const auto file = BB_FILES[f + (f == 0) - (f == 7)];
        const auto rank = BB_RANKS[r + (r == 0) - (r == 7)];

        return(file | shift_left(file) | shift_right(file))
            & (rank | shift_up(rank) | shift_down(rank))
            & ~BB_SQUARES[king];
    }


    /*
     * Count friendly pieces minus opponent's pieces in the king
     * quadrant, as an approximate measure of the king's safety.
     */
    static INLINE int eval_king_quadrant(const State& state, int pcs)
    {
        int score = 0;

        for (auto color: {BLACK, WHITE})
        {
            const auto ours = state.occupied_co(color);
            const auto theirs = state.occupied_co(!color);
            const auto king_mask = state.kings & ours;

            for (auto q : BB_QUANDRANTS)
            {
                if (king_mask & q)
                {
                    q &= ~king_mask;

                    score += SIGN[color] * (
                        popcount(q & ours) - popcount(q & theirs)
                        /* count queens as 2 pieces */
                        + popcount(q & state.queens & ours)
                        - popcount(q & state.queens & theirs));

                    break;
                }
            }
        }

        return score * interpolate(pcs, MIDGAME_KING_QUADRANT, ENDGAME_KING_QUADRANT);
    }


    static INLINE int eval_king_safety(const State& state, int pcs)
    {
        int attacks = 0; /* attacks around the king */
        int castle  = 0; /* castling rights bonuses */
        int outside = 0; /* penalty for king out of ranks [0,1] */
        int shield  = 0; /* pawn shield bonus */

        const auto occupied = state.occupied();

        for (auto color : { BLACK, WHITE })
        {
            const auto king = state.king(color);
            const auto ranks = color ? square_rank(king) : 7 - square_rank(king);
            const auto area = king_area(state, king);
            const auto color_mask = state.occupied_co(color);

            if (ranks > 1)
            {
                outside += SIGN[color] * (ranks - 1);
            }
            else
            {
                castle += SIGN[color]
                    * (ranks == 0)
                    * popcount(state.castling_rights & BB_BACKRANKS[color]);

                if (const auto pawns = state.pawns & color_mask)
                {
                    shield += SIGN[color]
                        * bool(BB_FILES[square_file(king)] & pawns)
                        * popcount(area & pawns);
                }
            }

            /*
             * https://www.chessprogramming.org/King_Safety  Attacking King Zone
             */
            static constexpr int ATTACK_WEIGHT[8] = { 0, 0, 50, 75, 88, 94, 97, 99 };

            for_each_square(area & ~color_mask, [&](Square square) {
                double attacks_value = 0;

                const auto attackers_mask =
                    state.attacker_pieces_mask(!color, square, occupied);

                for_each_square(attackers_mask, [&](Square attacking_square) {
                    const auto pt = state.piece_type_at(attacking_square);
                    ASSERT(pt > PieceType::PAWN && pt < PieceType::KING);

                    attacks_value += double(WEIGHT[pt] - 100) / KING_ATTACK_DIV;
                });

                const auto attackers = std::min(popcount(attackers_mask), 7);
                attacks -= SIGN[color] * attacks_value * ATTACK_WEIGHT[attackers] / 100;
            });
        }

        return attacks
            + eval_king_quadrant(state, pcs)
            + castle * interpolate(pcs, CASTLING_RIGHTS_BONUS, 0)
            + outside * interpolate(pcs, KING_OUT_PENALTY, 0)
            + shield * interpolate(pcs, PAWN_SHIELD, 0);
    }


    static INLINE int eval_material_imbalance(const State& state)
    {
        if ((state.rooks | state.queens) == 0)
        {
            const int pawn_cnt[] = {
                popcount(state.pawns & state.occupied_co(BLACK)),
                popcount(state.pawns & state.occupied_co(WHITE))
            };

            for (auto side : { BLACK, WHITE })
            {
                if (pawn_cnt[side] < pawn_cnt[!side]
                    && ((state.bishops | state.knights) & state.occupied_co(side))
                   )
                {
                    return SIGN[side] * MATERIAL_IMBALANCE;
                }
            }
        }
        return 0;
    }


    /*
     * If the difference in material is within a pawn or less, favor
     * the side with two minor pieces over the side with extra rook.
     */
    static INLINE int eval_redundant_rook(const State& state, int pcs)
    {
        int score = 0;

        for (auto color : { BLACK, WHITE })
        {
            score += SIGN[color]
                * (popcount((state.bishops | state.knights) & state.occupied_co(!color)) >= 2)
                * (popcount(state.rooks & state.occupied_co(color)) >= 2);
        }

        return score * interpolate(pcs, 0, REDUNDANT_ROOK);
    }


    static INLINE int eval_open_files(const State& state, int piece_count)
    {
        static constexpr Bitboard opposite_backranks[] = {
            BB_RANK_2 | BB_RANK_1,
            BB_RANK_7 | BB_RANK_8,
        };

        int open_score = 0;
        int half_open_score = 0;

        for (auto color : { BLACK, WHITE })
        {
            const auto s = SIGN[color];
            const auto own_color_mask = state.occupied_co(color);

            for (const auto file_mask : BB_FILES)
            {
                if (auto mask = file_mask & (state.rooks | state.queens) & own_color_mask)
                {
                    if ((file_mask & state.pawns) == BB_EMPTY)
                    {
                        open_score += s;
                    }
                    else
                        if ((file_mask & state.pawns & own_color_mask) == BB_EMPTY)
                        {
                            half_open_score += s;
                        }
                }
            }
        }
        return half_open_score * interpolate(piece_count, MIDGAME_HALF_OPEN_FILE, 0)
             + open_score * interpolate(piece_count, MIDGAME_OPEN_FILE, 0);
    }


    static INLINE int eval_passed_formation(const State& state, int piece_count)
    {
        const auto diff =
            state.longest_pawn_sequence(state.occupied_co(WHITE) & (BB_RANK_6 | BB_RANK_7)) -
            state.longest_pawn_sequence(state.occupied_co(BLACK) & (BB_RANK_2 | BB_RANK_3));

        return diff * interpolate(piece_count, MIDGAME_PASSED_FORMATION, ENDGAME_PASSED_FORMATION);
    }


    static INLINE Bitboard pawn_defenders(const State& state, Color color, Square square)
    {
        return state.occupied_co(color) & state.pawns & BB_PAWN_ATTACKS[!color][square];
    }


    static INLINE int eval_pawn_chain(const State& state, Color color, Square pawn, int (&pawn_chain_evals)[64])
    {
        int& score = pawn_chain_evals[pawn];

        if (!is_valid(score))
        {
            score = 0;

            if (const auto defenders = pawn_defenders(state, color, pawn))
            {
                for_each_square(defenders, [&](Square square) {
                    score += 1 + eval_pawn_chain(state, color, square, pawn_chain_evals);
                });
            }
        }
        return score;
    }


    template<int i>
    static INLINE int eval_passed_pawns(const State& state, int piece_count, int (&pawn_chain_evals)[2][64])
    {
        struct PassedPawnRank
        {
            Bitboard mask[2]; /* midgame, endgame */
            int bonus[2]; /* ditto */
        };

        static
    #if TUNING_ENABLED || TUNING_PARTIAL
            const
    #else
            constexpr
    #endif
        PassedPawnRank ranks[2] = {
            {   /* 6th rank */
                { BB_RANK_3, BB_RANK_6 },
                { MIDGAME_UNBLOCKED_PASSED_6, ENDGAME_UNBLOCKED_PASSED_6 }
            },
            {   /* 7th rank */
                { BB_RANK_2, BB_RANK_7 },
                { MIDGAME_UNBLOCKED_PASSED_7, ENDGAME_UNBLOCKED_PASSED_7 }
            },
        };

        int score = 0;
        const auto occupied = state.occupied();

        for (auto color : { BLACK, WHITE })
        {
            const auto sign = SIGN[color];
            const auto own_mask = state.occupied_co(color);
            const auto others_mask = state.occupied_co(!color);

            const auto pawns = state.pawns & own_mask & ranks[i].mask[color];

            for_each_square(pawns, [&](Square square) {
                const auto pawn_mask = BB_SQUARES[square];
                const auto advance_mask = color ? shift_up(pawn_mask) : shift_down(pawn_mask);

                if ((advance_mask & occupied) == 0)
                {
                    score += sign * interpolate(piece_count, ranks[i].bonus[0], ranks[i].bonus[1]);
                }
                else if (BB_PAWN_ATTACKS[color][square] & others_mask)
                {
                    score += sign * interpolate(piece_count, ranks[i].bonus[0], ranks[i].bonus[1]);
                }

                score += sign
                    * eval_pawn_chain(state, color, square, pawn_chain_evals[color])
                    * interpolate(piece_count, MIDGAME_DEFENDED_PASSED, ENDGAME_DEFENDED_PASSED);
            });
        }
        return score;
    }


    static INLINE int eval_pawn_structure(const State& state, int pc)
    {
        int eval = eval_passed_formation(state, pc);

        int pawn_chain_evals[2][64];
        std::fill_n(&pawn_chain_evals[0][0], 2 * 64, SCORE_MIN);

        eval += eval_passed_pawns<0>(state, pc, pawn_chain_evals);
        eval += eval_passed_pawns<1>(state, pc, pawn_chain_evals);

        int doubled = 0;
        int isolated = 0;
        int diff = 0;

        for (auto color : { BLACK, WHITE })
        {
            const auto sign = SIGN[color];

            if (const auto own_pawns = state.pawns & state.occupied_co(color))
            {
                for (const auto& bb_file : BB_FILES)
                {
                    auto n = popcount(own_pawns & bb_file);
                    doubled += sign * (n > 1) * (n - 1);
                }
                diff += sign * popcount(own_pawns);
                isolated += sign * state.count_isolated_pawns(color);
            }
        }

        return eval
            + doubled * interpolate(pc, MIDGAME_DOUBLED_PAWNS, ENDGAME_DOUBLED_PAWNS)
            + isolated * interpolate(pc, MIDGAME_ISOLATED_PAWNS, ENDGAME_ISOLATED_PAWNS)
            + (diff != 0) * SIGN[diff > 0] * interpolate(pc, MIDGAME_PAWN_MAJORITY, ENDGAME_PAWN_MAJORITY);
    }


    static INLINE int eval_threats(const State& state, int piece_count)
    {
        int diff = 0;
        const auto occupied = state.occupied();

        for (auto color : { BLACK, WHITE })
        {
            const auto sign = SIGN[color];
            for_each_square(state.occupied_co(color) & ~(state.pawns | state.kings), [&](Square square) {
                diff -= sign * popcount(state.attackers_mask(!color, square, occupied));
            });
        }
        return diff * interpolate(piece_count, MIDGAME_THREATS, ENDGAME_THREATS);
    }


    static INLINE int eval_tactical(const State& state, score_t mat_eval, int piece_count)
    {
        score_t eval = eval_center(state, piece_count);

        if (abs(mat_eval) < WEIGHT[PAWN])
        {
            eval += eval_material_imbalance(state);
            eval += eval_redundant_rook(state, piece_count);
        }

        eval += eval_open_files(state, piece_count);
        eval += eval_pawn_structure(state, piece_count);
        eval += eval_piece_grading(state, piece_count);
        eval += state.diff_connected_rooks()
             * interpolate(piece_count, MIDGAME_CONNECTED_ROOKS, ENDGAME_CONNECTED_ROOKS);

        if (state.bishops)
        {
            eval += BISHOP_PAIR * state.diff_bishop_pairs();
        }
        eval += eval_threats(state, piece_count);
        eval += eval_king_safety(state, piece_count);

        return eval;
    }


    static INLINE int eval_tactical(Context& ctxt, score_t eval)
    {
        const auto& state = ctxt.state();
        const auto piece_count = ctxt.piece_count();

        /*
         * 2nd order evaluation is currently slow (and possibly inaccurate).
         * To mitigate, in midgame it is done only at lower plies, and only
         * if 1st order eval delta is within a 2-3 pawns margin. The idea is
         * that deeper search paths may not benefit as much from qualitative
         * positional evaluation anyway; and tactical advantages will rarely
         * overcome significant material deficits.
         */
        if (state.is_endgame() || (ctxt._ply < EVAL_LOW_DEPTH && abs(eval) < EVAL_MARGIN))
        {
            const auto mat_eval = ctxt.evaluate_material();

            eval += SIGN[state.turn] * eval_tactical(state, mat_eval, piece_count);
        }
        else
        {
            eval += SIGN[state.turn] * eval_piece_grading(state, piece_count);
        }

        ASSERT(eval < SCORE_MAX);

        return eval;
    }
#endif /* !WITH_NNUE */

} /* namespace */

