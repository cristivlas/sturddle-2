#ifndef TBCONFIG_H
#define TBCONFIG_H

#include "chess.h"

#define TB_ASSERT ASSERT

/****************************************************************************/
/* BUILD CONFIG:                                                            */
/****************************************************************************/

/*
 * Define TB_CUSTOM_POP_COUNT to override the internal popcount
 * implementation. To do this supply a macro or function definition
 * here:
 */
#define TB_CUSTOM_POP_COUNT(x) static_cast<unsigned int>(chess::popcount(x))

/*
 * Define TB_CUSTOM_LSB to override the internal lsb
 * implementation. To do this supply a macro or function definition
 * here:
 */
#define TB_CUSTOM_LSB(x) chess::lsb(x)

/*
 * Define TB_NO_STDINT if you do not want to use <stdint.h> or it is not
 * available.
 */
/* #define TB_NO_STDINT */

/*
 * Define TB_NO_STDBOOL if you do not want to use <stdbool.h> or it is not
 * available or unnecessary (e.g. C++).
 */
/* #define TB_NO_STDBOOL */

/*
 * Define TB_NO_THREADS if your program is not multi-threaded.
 */
/* #define TB_NO_THREADS */

/*
 * Define TB_NO_HELPER_API if you do not need the helper API.
 */
/* #define TB_NO_HELPER_API */

/*
 * Define TB_NO_HW_POP_COUNT if there is no hardware popcount instruction.
 *
 * Note: if defined, TB_CUSTOM_POP_COUNT is always used in preference
 * to any built-in popcount functions.
 *
 * If no custom popcount function is defined, and if the following
 * define is not set, the code will attempt to use an available hardware
 * popcnt (currently supported on x86_64 architecture only) and otherwise
 * will fall back to a software implementation.
 */
/* #define TB_NO_HW_POP_COUNT */

/***************************************************************************/
/* SCORING CONSTANTS                                                       */
/***************************************************************************/
/*
 * Fathom can produce scores for tablebase moves. These depend on the
 * value of a pawn, and the magnitude of mate scores. The following
 * constants are representative values but will likely need
 * modification to adapt to an engine's own internal score values.
 */
#define TB_VALUE_PAWN chess::WEIGHT[chess::PAWN]  /* value of pawn in endgame */
#define TB_VALUE_MATE MATE_HIGH
#define TB_VALUE_INFINITE SCORE_MAX /* value above all normal score values */
#define TB_VALUE_DRAW 0
#define TB_MAX_MATE_PLY PLY_MAX

/***************************************************************************/
/* ENGINE INTEGRATION CONFIG                                               */
/***************************************************************************/
#if true
/*
 * If you are integrating tbprobe into an engine, you can replace some of
 * tbprobe's built-in functionality with that already provided by the engine.
 * This is OPTIONAL.  If no definition are provided then tbprobe will use its
 * own internal defaults.  That said, for engines it is generally a good idea
 * to avoid redundancy.
 */

/*
 * Define TB_KING_ATTACKS(square) to return the king attacks bitboard for a
 * king at `square'.
 */
#define TB_KING_ATTACKS(square) chess::BB_KING_ATTACKS[square]

/*
 * Define TB_KNIGHT_ATTACKS(square) to return the knight attacks bitboard for
 * a knight at `square'.
 */
#define TB_KNIGHT_ATTACKS(square) chess::BB_KNIGHT_ATTACKS[square]

/*
 * Define TB_ROOK_ATTACKS(square, occ) to return the rook attacks bitboard
 * for a rook at `square' assuming the given `occ' occupancy bitboard.
 */
/* #define TB_ROOK_ATTACKS(square, occ)     <DEFINITION> */

/*
 * Define TB_BISHOP_ATTACKS(square, occ) to return the bishop attacks bitboard
 * for a bishop at `square' assuming the given `occ' occupancy bitboard.
 */
/* #define TB_BISHOP_ATTACKS(square, occ)   <DEFINITION> */

/*
 * Define TB_QUEEN_ATTACKS(square, occ) to return the queen attacks bitboard
 * for a queen at `square' assuming the given `occ' occupancy bitboard.
 * NOTE: If no definition is provided then tbprobe will use:
 *       TB_ROOK_ATTACKS(square, occ) | TB_BISHOP_ATTACKS(square, occ)
 */
/* #define TB_QUEEN_ATTACKS(square, occ)    <DEFINITION> */

#if USE_MAGIC_BITS
#define TB_ROOK_ATTACKS(sq, occ_mask)   magic_bits_attacks.Rook(occ_mask, sq)
#define TB_BISHOP_ATTACKS(sq, occ_mask) magic_bits_attacks.Bishop(occ_mask, sq)
#else
#define TB_ROOK_ATTACKS(sq, occ_mask)   chess::BB_ROOK_ATTACKS.get(sq, occ_mask)
#define TB_BISHOP_ATTACKS(sq, occ_mask) chess::BB_DIAG_ATTACKS.get(sq, occ_mask)
#endif /* USE_MAGIC_BITS */

/*
 * Define TB_PAWN_ATTACKS(square, color) to return the pawn attacks bitboard
 * for a `color' pawn at `square'.
 * NOTE: This definition must work for pawns on ranks 1 and 8.  For example,
 *       a white pawn on e1 attacks d2 and f2.  A black pawn on e1 attacks
 *       nothing.  Etc.
 * NOTE: This definition must not include en passant captures.
 */
#define TB_PAWN_ATTACKS(square, color) chess::BB_PAWN_ATTACKS[color][square]
#endif

#endif
