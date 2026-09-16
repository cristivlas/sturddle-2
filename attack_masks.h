#pragma once
/*
 * Attack planes in hidden_1c column order, computed from scratch at eval.
 */
#include "chess.h"

#if ATTACK_MASKS

namespace chess
{
    struct AttackMaskSet
    {
        Bitboard _cols[12] = { }; /* (king, pawn, knight, bishop, rook, queen) x (black, white) */

        void compute(const State& state)
        {
            const auto occupied = state.occupied();
            _cols[0] = king(state, BLACK);
            _cols[1] = king(state, WHITE);
            _cols[2] = pawns<BLACK>(state);
            _cols[3] = pawns<WHITE>(state);
            _cols[4] = knights(state, BLACK);
            _cols[5] = knights(state, WHITE);
            _cols[6] = sliders<BISHOP>(state, BLACK, occupied);
            _cols[7] = sliders<BISHOP>(state, WHITE, occupied);
            _cols[8] = sliders<ROOK>(state, BLACK, occupied);
            _cols[9] = sliders<ROOK>(state, WHITE, occupied);
            _cols[10] = sliders<QUEEN>(state, BLACK, occupied);
            _cols[11] = sliders<QUEEN>(state, WHITE, occupied);
        }

    private:
        static INLINE Bitboard king(const State& state, Color c)
        {
            const auto bb = state.pieces_mask(KING, c);
            return bb ? BB_KING_ATTACKS[lsb(bb)] : BB_EMPTY;
        }

        template <Color C>
        static INLINE Bitboard pawns(const State& state)
        {
            const auto bb = state.pieces_mask(PAWN, C);
            if constexpr (C == WHITE)
                return ((bb << 7) & ~BB_FILES[7]) | ((bb << 9) & ~BB_FILE_A);
            else
                return ((bb >> 7) & ~BB_FILE_A) | ((bb >> 9) & ~BB_FILES[7]);
        }

        static INLINE Bitboard knights(const State& state, Color c)
        {
            auto mask = BB_EMPTY;
            for_each_square_r(state.pieces_mask(KNIGHT, c), [&](Square sq) { mask |= BB_KNIGHT_ATTACKS[sq]; });
            return mask;
        }

        template <PieceType T>
        static INLINE Bitboard sliders(const State& state, Color c, Bitboard occupied)
        {
            auto mask = BB_EMPTY;
            for_each_square_r(state.pieces_mask(T, c), [&](Square sq) {
                if constexpr (T != ROOK)
                    mask |= diagonal_attacks(occupied, sq);
                if constexpr (T != BISHOP)
                    mask |= rank_and_file_attacks(occupied, sq);
            });
            return mask;
        }
    };
}

#endif /* ATTACK_MASKS */
