/*
 * Sturddle Chess Engine (C) 2022, 2023 Cristian Vlasceanu
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
#pragma once

#include "common.h"
/*
 * Piece-square tables.
 * https://www.chessprogramming.org/Simplified_Evaluation_Function
 */
#if USE_PIECE_SQUARE_TABLES

static
#if !PS_PAWN_TUNING_ENABLED && !PS_KNIGHT_TUNING_ENABLED && !PS_BISHOP_TUNING_ENABLED && \
    !PS_ROOK_TUNING_ENABLED && !PS_QUEEN_TUNING_ENABLED && !PS_KING_TUNING_ENABLED
    constexpr
#endif

int SQUARE_TABLE[][64] = {
    {}/* NONE */,
    { /* PAWN */
          0,   0,   0,   0,   0,   0,  0,   0,
         24,  33,  15,  24,  17,  31,  8,  -3,
         -1,   2,   6,   8,  16,  14,  6,  -5,
         -3,   3,   1,   5,   6,   3,  4,  -6,
         -7,   0,  -1,   3,   4,   1,  2,  -6,
         -6,  -1,  -1,  -2,   1,   1,  8,  -3,
         -9,   0,  -5,  -6,  -4,   6, 10,  -5,
          0,   0,   0,   0,   0,   0,  0,   0,
    },
    { /* KNIGHT */
        -35, -22,  -8, -12,  15, -24,  -4, -27,
        -18, -10,  18,   9,   6,  15,   2,  -4,
        -12,  15,   9,  16,  21,  32,  18,  11,
         -2,   4,   5,  13,   9,  17,   4,   5,
         -3,   1,   4,   3,   7,   5,   5,  -2,
         -6,  -2,   3,   2,   5,   4,   6,  -4,
         -7, -13,  -3,  -1,   0,   4,  -3,  -5,
        -26,  -5, -14,  -8,  -4,  -7,  -5,  -6,
    },
    { /* BISHOP */
         -7,   1, -20,  -9,  -6, -10,   2,  -2,
         -6,   4,  -4,  -3,   7,  15,   4, -12,
         -4,   9,  11,  10,   9,  12,   9,   0,
         -1,   1,   5,  12,   9,   9,   2,   0,
         -1,   3,   3,   6,   8,   3,   2,   1,
          0,   4,   4,   4,   3,   7,   4,   2,
          1,   4,   4,   0,   2,   5,   8,   0,
         -8,  -1,  -3,  -5,  -3,  -3, -10,  -5,
    },
    { /* ROOK */
          8,  10,   8,  13, 16,   2,   8,  11,
          7,   8,  14,  15, 20,  17,   6,  11,
         -1,   5,   6,   9,  4,  11,  15,   4,
         -6,  -3,   2,   6,  6,   9,  -2,  -5,
         -9,  -6,  -3,   0,  2,  -2,   1,  -6,
        -11,  -6,  -4,  -4,  1,   0,  -1,  -8,
        -11,  -4,  -5,  -2,  0,   3,  -1, -18,
         -5,  -3,   0,   4,  4,   2,  -9,  -6,
    },
    { /* QUEEN */
         -7,   0,   7,   3,  15,  11,  11,  11,
         -6, -10,  -1,   0,  -4,  14,   7,  13,
         -3,  -4,   2,   2,   7,  14,  12,  14,
         -7,  -7,  -4,  -4,   0,   4,   0,   0,
         -2,  -6,  -2,  -2,   0,  -1,   1,  -1,
         -3,   0,  -3,   0,  -1,   0,   3,   1,
         -9,  -2,   3,   0,   2,   4,  -1,   0,
          0,  -4,  -2,   2,  -4,  -6,  -8, -12,
    },
    { /* KING */
        -16,   6,   4,  -4, -14,  -8,   0,   3,
          7,   0,  -5,  -2,  -2,  -1, -10,  -7,
         -2,   6,   0,  -4,  -5,   1,   5,  -5,
         -4,  -5,  -3,  -7,  -8,  -6,  -3,  -9,
        -12,   0,  -7, -10, -12, -11,  -8, -13,
         -3,  -3,  -5, -12, -11,  -8,  -4,  -7,
          0,   2,  -2, -16, -11,  -4,   2,   2,
         -4,   9,   3, -14,   2,  -7,   6,   3,
    }
};


static
#if !PS_KING_TUNING_ENABLED
    constexpr
#endif
int ENDGAME_KING_SQUARE_TABLE[64] = {
    -18,  -9,  -4,  -4,  -3,   4,   1,  -4,
     -3,   4,   3,   4,   4,  10,   6,   3,
      2,   4,   6,   4,   5,  11,  11,   3,
     -2,   5,   6,   7,   6,   8,   6,   1,
     -4,  -1,   5,   6,   7,   6,   2,  -3,
     -5,  -1,   3,   5,   6,   4,   2,  -2,
     -7,  -3,   1,   3,   3,   1,  -1,  -4,
    -13,  -8,  -5,  -3,  -7,  -3,  -6, -11,
};

#endif /* USE_PIECE_SQUARE_TABLES */
