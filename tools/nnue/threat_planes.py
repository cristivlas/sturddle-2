#! /usr/bin/env python3
"""
Batch computation of per-piece-type attack planes from packed bitboards.

Input/output column layout matches the H5 feature encoding: 12 uint64 columns,
(king, pawn, knight, bishop, rook, queen) x (black, white). Pure numpy on
np.uint64; sliders use Kogge-Stone occluded fills. Shared by the TF and torch
trainers; must be applied after any color-flip augmentation.

Self-test against python-chess:

    python tools/nnue/threat_planes.py
"""

import numpy as np

U64 = np.uint64

PLANES = 12  # one attack plane per (piece type, color)
PACKED_INPUTS = 13 + PLANES  # model input columns with threat planes appended

NOT_FILE_A = U64(0xFEFEFEFEFEFEFEFE)
NOT_FILE_H = U64(0x7F7F7F7F7F7F7F7F)
NOT_FILE_AB = U64(0xFCFCFCFCFCFCFCFC)
NOT_FILE_GH = U64(0x3F3F3F3F3F3F3F3F)

KING, PAWN, KNIGHT, BISHOP, ROOK, QUEEN = 0, 2, 4, 6, 8, 10  # column base per type; +color for black/white


def _shl(b, n):
    return b << U64(n)


def _shr(b, n):
    return b >> U64(n)


def _occl_attacks(gen, pro, n, wrap, left):
    """Kogge-Stone occluded fill + final shift: attacks of sliders `gen` through empty `pro` in one direction."""
    sh = _shl if left else _shr
    pro = pro & wrap
    gen = gen | (pro & sh(gen, n))
    pro = pro & sh(pro, n)
    gen = gen | (pro & sh(gen, 2 * n))
    pro = pro & sh(pro, 2 * n)
    gen = gen | (pro & sh(gen, 4 * n))
    return sh(gen, n) & wrap


ALL = U64(0xFFFFFFFFFFFFFFFF)
ORTH_DIRS = ((8, ALL, True), (8, ALL, False), (1, NOT_FILE_A, True), (1, NOT_FILE_H, False))
DIAG_DIRS = ((9, NOT_FILE_A, True), (7, NOT_FILE_H, True), (7, NOT_FILE_A, False), (9, NOT_FILE_H, False))


def _slider_attacks(bb, empty, dirs):
    attacks = np.zeros_like(bb)
    for n, wrap, left in dirs:
        attacks |= _occl_attacks(bb, empty, n, wrap, left)
    return attacks


def _pawn_attacks(bb, white):
    if white:
        return (_shl(bb, 7) & NOT_FILE_H) | (_shl(bb, 9) & NOT_FILE_A)
    return (_shr(bb, 7) & NOT_FILE_A) | (_shr(bb, 9) & NOT_FILE_H)


def _knight_attacks(bb):
    return (
        (_shl(bb, 17) & NOT_FILE_A)
        | (_shl(bb, 15) & NOT_FILE_H)
        | (_shl(bb, 10) & NOT_FILE_AB)
        | (_shl(bb, 6) & NOT_FILE_GH)
        | (_shr(bb, 17) & NOT_FILE_H)
        | (_shr(bb, 15) & NOT_FILE_A)
        | (_shr(bb, 10) & NOT_FILE_GH)
        | (_shr(bb, 6) & NOT_FILE_AB)
    )


def _king_attacks(bb):
    return (
        (_shl(bb, 1) & NOT_FILE_A)
        | (_shr(bb, 1) & NOT_FILE_H)
        | (_shl(bb, 9) & NOT_FILE_A)
        | (_shl(bb, 7) & NOT_FILE_H)
        | (_shr(bb, 7) & NOT_FILE_A)
        | (_shr(bb, 9) & NOT_FILE_H)
        | _shl(bb, 8)
        | _shr(bb, 8)
    )


def append_planes(x):
    """x: (B, 13) uint64 [12 bitboards + turn] -> (B, 25) with the 12 attack planes inserted before turn."""
    return np.concatenate([x[:, :12], attack_planes(x).astype(x.dtype), x[:, 12:]], axis=1)


def attack_planes(bitboards):
    """bitboards: (B, 12) uint64 piece columns -> (B, 12) uint64 attack planes, same column layout."""
    bb = np.ascontiguousarray(bitboards[:, :12], dtype=U64)
    empty = ~np.bitwise_or.reduce(bb, axis=1)

    planes = np.empty_like(bb)
    for color in (0, 1):
        planes[:, KING + color] = _king_attacks(bb[:, KING + color])
        planes[:, PAWN + color] = _pawn_attacks(bb[:, PAWN + color], white=bool(color))
        planes[:, KNIGHT + color] = _knight_attacks(bb[:, KNIGHT + color])
        planes[:, BISHOP + color] = _slider_attacks(bb[:, BISHOP + color], empty, DIAG_DIRS)
        planes[:, ROOK + color] = _slider_attacks(bb[:, ROOK + color], empty, ORTH_DIRS)
        planes[:, QUEEN + color] = _slider_attacks(bb[:, QUEEN + color], empty, ORTH_DIRS + DIAG_DIRS)
    return planes


def _self_test():
    import random

    import chess

    from golds import TESTS

    type_to_col = {
        chess.KING: KING,
        chess.PAWN: PAWN,
        chess.KNIGHT: KNIGHT,
        chess.BISHOP: BISHOP,
        chess.ROOK: ROOK,
        chess.QUEEN: QUEEN,
    }

    def encode(board):
        mask = [board.occupied_co[chess.BLACK], board.occupied_co[chess.WHITE]]
        pieces = (board.kings, board.pawns, board.knights, board.bishops, board.rooks, board.queens)
        return np.asarray([[pcs & m for m in mask] for pcs in pieces], dtype=U64).ravel()

    def expected_planes(board):
        expected = np.zeros(12, dtype=U64)
        for sq, piece in board.piece_map().items():
            expected[type_to_col[piece.piece_type] + int(piece.color)] |= U64(int(board.attacks(sq)))
        return expected

    boards = [chess.Board(fen) for fen in TESTS]
    rng = random.Random(44)
    for _ in range(500):
        board = chess.Board()
        for _ in range(rng.randrange(1, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        boards.append(board)

    batch = np.stack([encode(b) for b in boards])
    planes = attack_planes(batch)
    for i, board in enumerate(boards):
        expected = expected_planes(board)
        if not np.array_equal(planes[i], expected):
            bad = [c for c in range(12) if planes[i][c] != expected[c]]
            raise AssertionError(f"mismatch at {board.fen()}, columns {bad}")
    print(f"OK: {len(boards)} positions, all 12 planes match python-chess")


if __name__ == "__main__":
    _self_test()
