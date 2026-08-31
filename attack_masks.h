#pragma once
/*
 * Incrementally maintained per-piece attack masks, with per-type and per-side
 * aggregate planes.
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
        Bitboard _piece[64] = { }; /* valid only where occupied */
        Bitboard _by_type[2][7] = { }; /* [color][PieceType] */
        Bitboard _by_side[2] = { };
        uint64_t _hash = 0;

        INLINE bool needs_update(const State& state) const
        {
            return state.hash() != _hash;
        }

        void full_rebuild(const State& state)
        {
            const auto occupied = state.occupied();
            for (auto c : { BLACK, WHITE })
            {
                const auto ours = state.occupied_co(c);
                _by_side[c] = BB_EMPTY;
                for (auto t : PIECES)
                {
                    auto mask = BB_EMPTY;
                    for_each_square(state.pieces(t) & ours, [&](Square sq) {
                        mask |= _piece[sq] = state.attacks_mask(sq, occupied);
                    });
                    _by_side[c] |= _by_type[c][t] = mask;
                }
            }
            _hash = state.hash();
        }

        void update(const AttackMaskSet& prev, const State& prev_state, const State& state, const Move& move)
        {
            ASSERT(this != &prev);
            ASSERT(prev._hash == prev_state.hash());

            *this = prev;
            _hash = state.hash();

            if (!move) /* null move: board unchanged */
            {
                debug_validate(state);
                return;
            }

            if (move.promotion() || prev_state.is_castling(move) || prev_state.is_en_passant(move))
            {
                full_rebuild(state);
                return;
            }

            const auto from = move.from_square();
            const auto to = move.to_square();
            const auto to_mask = BB_SQUARES[to];
            const auto touched = BB_SQUARES[from] | to_mask;
            const auto occupied = state.occupied();
            const auto color = prev_state.turn;
            const auto piece_type = prev_state.piece_type_at(from);
            const auto capture_type = prev_state.piece_type_at(to);

            bool dirty[2][7] = { };
            dirty[color][piece_type] = true;
            if (capture_type)
                dirty[!color][capture_type] = true;

            _piece[from] = BB_EMPTY;
            _piece[to] = state.attacks_mask(to, occupied);

            /* only sliders with rays through the touched squares see the occupancy change */
            for_each_square((state.bishops | state.rooks | state.queens) & ~to_mask, [&](Square sq) {
                if (_piece[sq] & touched)
                {
                    _piece[sq] = state.attacks_mask(sq, occupied);
                    dirty[state.piece_color_at(sq)][state.piece_type_at(sq)] = true;
                }
            });

            for (auto c : { BLACK, WHITE })
            {
                bool dirty_side = false;
                for (auto t : PIECES)
                    if (dirty[c][t])
                    {
                        _by_type[c][t] = plane(state, t, c);
                        dirty_side = true;
                    }
                if (dirty_side)
                    _by_side[c] = _by_type[c][PAWN] | _by_type[c][KNIGHT] | _by_type[c][BISHOP]
                        | _by_type[c][ROOK] | _by_type[c][QUEEN] | _by_type[c][KING];
            }
            debug_validate(state);
        }

    private:
        /* Aggregate one (type, color) plane; pawns via whole-board shifts,
         * knights/kings via tables, sliders from the per-piece masks.
         */
        INLINE Bitboard plane(const State& state, PieceType t, Color c) const
        {
            const auto bb = state.pieces_mask(t, c);
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
            default:
                for_each_square(bb, [&](Square sq) { mask |= _piece[sq]; });
                return mask;
            }
        }

        INLINE void debug_validate(const State& state) const
        {
        #if DEBUG_INCREMENTAL
            AttackMaskSet temp;
            temp.full_rebuild(state);

            for_each_square(state.occupied(), [&](Square sq) {
                ASSERT_ALWAYS(_piece[sq] == temp._piece[sq]);
            });
            for (auto c : { BLACK, WHITE })
            {
                ASSERT_ALWAYS(_by_side[c] == temp._by_side[c]);
                for (auto t : PIECES)
                    ASSERT_ALWAYS(_by_type[c][t] == temp._by_type[c][t]);
            }
        #endif /* DEBUG_INCREMENTAL */
        }
    };
}

#endif /* ATTACK_MASKS */
