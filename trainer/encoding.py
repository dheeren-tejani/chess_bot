"""Python mirrors of the C++ engine's encoding functions.

These MUST stay byte-compatible with:
    engine/src/policy_map.h/.cpp   (move <-> action index)
    engine/src/encoder.h/.cpp      (position -> input planes/scalars)

Parity is enforced by tests/test_parity.py.
"""
from __future__ import annotations

import chess  # python-chess

NUM_PLANES = 113          # 112 age-block planes + 1 en-passant plane
HIST_STEPS = 8
PLANES_PER_AGE = 14
AGE_BLOCK_PLANES = HIST_STEPS * PLANES_PER_AGE   # 112
EP_PLANE = 112
SCALAR_COUNT = 9
POLICY_ACTIONS = 73 * 64

# Queen direction order: N, NE, E, SE, S, SW, W, NW (relative to mover)
QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# Knight delta order (df, dr) - must match types.h KNIGHT_DELTAS
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

# sq index convention: a1=0 .. h8=63 ; python-chess square = rank*8+file with a1=0 ✓


def _dir_index(df: int, rel_dr: int) -> int:
    if rel_dr > 0:
        return 0 if df == 0 else (1 if df > 0 else 7)
    if rel_dr < 0:
        return 4 if df == 0 else (3 if df > 0 else 5)
    return 2 if df > 0 else 6


def _knight_plane(df: int, rel_dr: int) -> int:
    return KNIGHT_DELTAS.index((df, rel_dr))


def move_policy_index(move: chess.Move, board: chess.Board) -> int:
    """Mirror of chess::move_policy_index."""
    f = move.from_square
    t = move.to_square
    df = chess.square_file(t) - chess.square_file(f)
    dr = chess.square_rank(t) - chess.square_rank(f)
    rel_dr = dr if board.turn == chess.WHITE else -dr
    dist = max(abs(df), abs(dr))

    promo = move.promotion
    if promo is not None and promo != chess.QUEEN:
        cap_dir = 0 if df < 0 else (2 if df > 0 else 1)
        p_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[promo]
        plane = 64 + p_idx * 3 + cap_dir
    elif (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        plane = 56 + _knight_plane(df, rel_dr)
    else:
        plane = _dir_index(df, rel_dr) * 7 + (dist - 1)
    return plane * 64 + f


def _pos_key(board: chess.Board) -> tuple:
    """Position identity matching C++ zobrist coverage (pieces, stm, castle, ep)."""
    return (
        board.pawns, board.knights, board.bishops, board.rooks,
        board.queens, board.kings, board.occupied,
        board.turn, board.castling_rights, board.ep_square,
    )


def encode_board(board: chess.Board, history: list[chess.Board] | None = None):
    """Mirror of chess::encode_position.

    `history` includes `board` as its LAST element when given.
    Returns (planes bytes-like np.uint64[112], scalars float32[4]).
    """
    import numpy as np

    hist = history if history is not None else [board]
    assert hist[-1] is board or hist[-1].fen() == board.fen()

    # repetition count of current position within game history
    key = _pos_key(board)
    seen_before = sum(1 for p in hist[:-1] if _pos_key(p) == key)
    rep_extra = seen_before  # occurrences excluding the position itself

    planes = np.zeros(NUM_PLANES, dtype=np.uint64)
    for age in range(HIST_STEPS):
        idx = len(hist) - 1 - age
        if idx < 0:
            continue
        b = hist[idx]
        base = age * PLANES_PER_AGE
        for sq in range(64):
            pc = b.piece_at(sq)
            if pc is None:
                continue
            plane = base + PIECE_TO_PLANE[(pc.color, pc.piece_type)]
            planes[plane] |= np.uint64(1 << sq)
        if age == 0:
            if rep_extra >= 1:
                planes[base + 12] = np.uint64((1 << 64) - 1)
            if rep_extra >= 2:
                planes[base + 13] = np.uint64((1 << 64) - 1)

    # en-passant target square (plane 112)
    if board.ep_square is not None:
        planes[EP_PLANE] = np.uint64(1 << board.ep_square)

    # NB: python-chess stores castling rights as a bitboard over the ROOK home
    # squares (a1,h1,a8,h8) - NOT the king destination squares.
    cr = board.castling_rights
    wk = bool(cr & chess.BB_H1)
    wq = bool(cr & chess.BB_A1)
    bk = bool(cr & chess.BB_H8)
    bq = bool(cr & chess.BB_A8)
    scalars = np.array([
        min(board.halfmove_clock / 100.0, 1.0),
        min(board.fullmove_number / 200.0, 1.0),
        min(rep_extra, 3) / 3.0,
        1.0 if board.turn == chess.WHITE else 0.0,
        1.0 if wk else 0.0,
        1.0 if wq else 0.0,
        1.0 if bk else 0.0,
        1.0 if bq else 0.0,
        1.0 if board.ep_square is not None else 0.0,
    ], dtype=np.float32)
    return planes, scalars


def material_diff_stm(board: chess.Board) -> int:
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    diff = 0
    for pt, v in vals.items():
        diff += v * len(board.pieces(pt, chess.WHITE))
        diff -= v * len(board.pieces(pt, chess.BLACK))
    return diff if board.turn == chess.WHITE else -diff
