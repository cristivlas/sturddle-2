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
    uint64_t attack_masks_checksum();

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
            rebuild(state, [&](Square sq, PieceType, Color) {
                return state.attacks_mask(sq, occupied);
            });
        }

        void update(const AttackMaskSet& prev, const State& prev_state, const State& state, const Move& move)
        {
            ASSERT(this != &prev);
            ASSERT(needs_update(state));
            ASSERT(prev._hash == prev_state.hash());

            if (!move) /* null move: board unchanged */
            {
                *this = prev;
                _hash = state.hash();
                debug_validate(state);
                return;
            }

            if (move.promotion() || prev_state.is_castling(move) || prev_state.is_en_passant(move))
            {
                full_rebuild(state);
                return;
            }

            const auto to_mask = BB_SQUARES[move.to_square()];
            const auto touched = BB_SQUARES[move.from_square()] | to_mask;
            const auto occupied = state.occupied();

            /* Non-sliders are plain table lookups; sliders reuse the previous
             * mask unless they moved or a ray reaches the changed squares.
             */
            rebuild(state, [&](Square sq, PieceType piece_type, Color color) {
                switch (piece_type)
                {
                case PAWN:
                    return BB_PAWN_ATTACKS[color][sq];
                case KNIGHT:
                    return BB_KNIGHT_ATTACKS[sq];
                case KING:
                    return BB_KING_ATTACKS[sq];
                default:
                    {
                        const auto mask = prev._piece[sq];
                        if ((BB_SQUARES[sq] & to_mask) || (mask & touched))
                            return state.attacks_mask(sq, occupied);
                        return mask;
                    }
                }
            });
            debug_validate(state);
        }

    private:
        /* Single fused pass: write per-piece masks and build all aggregate
         * planes, visiting only occupied squares.
         */
        template <typename F> INLINE void rebuild(const State& state, F mask_of)
        {
            for (auto c : { BLACK, WHITE })
            {
                const auto ours = state.occupied_co(c);
                _by_side[c] = BB_EMPTY;
                for (auto t : PIECES)
                {
                    auto plane = BB_EMPTY;
                    for_each_square(state.pieces(t) & ours, [&](Square sq) {
                        plane |= _piece[sq] = mask_of(sq, t, c);
                    });
                    _by_side[c] |= _by_type[c][t] = plane;
                }
            }
            _hash = state.hash();
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
