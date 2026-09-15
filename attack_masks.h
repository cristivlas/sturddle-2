#pragma once
/*
 * Incrementally maintained per-(color, type) attack planes.
 */
#include "chess.h"

#ifndef DEBUG_INCREMENTAL
    #define DEBUG_INCREMENTAL false
#endif /* DEBUG_INCREMENTAL */

#if ATTACK_MASKS

namespace chess
{
    struct AttackMaskSet
    {
        Bitboard _by_type[2][7] = { }; /* [color][PieceType] */
        uint64_t _hash = 0;

        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != _hash;
        }

        void full_rebuild(const State& state)
        {
            for (auto c : { BLACK, WHITE })
                for (auto t : PIECES)
                    _by_type[c][t] = plane(state, t, c);
            _hash = state.hash();
        }

        /* in-place: *this must hold prev_state's planes */
        void update(const State& prev_state, const State& state, const Move& move)
        {
            ASSERT(_hash == prev_state.hash());

            _hash = state.hash();

            if (!move) /* null move: board unchanged */
            {
                debug_validate(state);
                return;
            }

            const auto occupied = state.occupied();
            /* captures leave `to` occupied on both sides of the diff */
            const auto touched = (prev_state.occupied() ^ occupied) | BB_SQUARES[move.to_square()];

            /* sliders whose rays cross a touched square are exactly those seen from the touched squares */
            auto orth = BB_EMPTY, diag = BB_EMPTY;
            for_each_square(touched, [&](Square sq) {
                orth |= rank_and_file_attacks(occupied, sq);
                diag |= diagonal_attacks(occupied, sq);
            });
            const auto affected =
                ((orth & (state.rooks | state.queens)) | (diag & (state.bishops | state.queens))) & ~touched;

            for (auto c : { BLACK, WHITE })
                for (auto t : PIECES)
                {
                    const auto mask = state.pieces_mask(t, c);
                    if (mask != prev_state.pieces_mask(t, c) || (affected & mask) != BB_EMPTY)
                        _by_type[c][t] = plane(state, t, c);
                }

            debug_validate(state);
        }

    private:
        /* Aggregate one (type, color) plane: pawns via shifts, knights/kings via tables, sliders via magics. */
        INLINE Bitboard plane(const State& state, PieceType t, Color c) const
        {
            const auto bb = state.pieces_mask(t, c);
            const auto occupied = state.occupied();
            auto mask = BB_EMPTY;
            switch (t)
            {
            case PAWN:
                return c == WHITE
                    ? (((bb << 7) & ~BB_FILES[7]) | ((bb << 9) & ~BB_FILE_A))
                    : (((bb >> 7) & ~BB_FILE_A) | ((bb >> 9) & ~BB_FILES[7]));
            case KNIGHT:
                for_each_square(bb, [&](Square sq) { mask |= BB_KNIGHT_ATTACKS[sq]; });
                return mask;
            case KING:
                for_each_square(bb, [&](Square sq) { mask |= BB_KING_ATTACKS[sq]; });
                return mask;
            case BISHOP:
                for_each_square(bb, [&](Square sq) { mask |= diagonal_attacks(occupied, sq); });
                return mask;
            case ROOK:
                for_each_square(bb, [&](Square sq) { mask |= rank_and_file_attacks(occupied, sq); });
                return mask;
            default: /* QUEEN */
                for_each_square(bb, [&](Square sq) {
                    mask |= diagonal_attacks(occupied, sq) | rank_and_file_attacks(occupied, sq);
                });
                return mask;
            }
        }

        INLINE void debug_validate(const State& state) const
        {
        #if DEBUG_INCREMENTAL
            AttackMaskSet temp;
            temp.full_rebuild(state);

            for (auto c : { BLACK, WHITE })
                for (auto t : PIECES)
                    ASSERT_ALWAYS(_by_type[c][t] == temp._by_type[c][t]);
        #endif /* DEBUG_INCREMENTAL */
        }
    };
}

#endif /* ATTACK_MASKS */
