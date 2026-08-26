"""C++ <-> Python parity tests: policy indices and input encoding."""
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trainer.encoding import (NUM_PLANES, encode_board, material_diff_stm,
                              move_policy_index)

ENGINE = Path(__file__).resolve().parent.parent / "engine" / "build" / "engine"

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 3 42",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R b KQ - 1 8",
    "7k/P7/8/8/8/8/6p1/K7 w - - 0 1",           # promotions incl. underpromo
    "8/8/8/8/8/2k5/2p5/K7 b - - 0 1",
]


def engine_lines(*args):
    out = subprocess.run([str(ENGINE), *args], capture_output=True, text=True,
                         check=True)
    return out.stdout.strip().splitlines()


def test_engine_exists():
    assert ENGINE.exists(), "build the engine first (cmake --build engine/build)"


@pytest.mark.parametrize("fen", FENS)
def test_policy_index_parity(fen):
    board = chess.Board(fen)
    ref = {}
    for mv in board.legal_moves:
        board.push(mv)
        # engine enumerates from same position; use uci key
        board.pop()
        ref[mv.uci()] = None
        break
    # full comparison via engine output
    eng = {}
    for line in engine_lines("policy-index", fen):
        parts = line.split()
        eng[parts[0]] = int(parts[1])
    for mv in board.legal_moves:
        uci = mv.uci()
        assert uci in eng, f"missing {uci}"
        mine = move_policy_index(mv, board)
        assert mine == eng[uci], f"{uci}: python={mine} engine={eng[uci]}"


@pytest.mark.parametrize("fen", FENS)
def test_encoder_parity(fen):
    board = chess.Board(fen)
    planes_py, scalars_py = encode_board(board)
    mat_py = material_diff_stm(board)

    out = engine_lines("encode", fen)
    assert len(out) == NUM_PLANES + 2, f"unexpected encode output lines: {len(out)}"
    words = [int(ln, 16) for ln in out[:NUM_PLANES]]
    scalars_cpp = np.array([float(x) for x in out[NUM_PLANES].split()], dtype=np.float32)
    mat_cpp = int(out[NUM_PLANES + 1])

    planes_cpp = np.zeros(NUM_PLANES, dtype=np.uint64)
    for i, w in enumerate(words):
        planes_cpp[i] = np.uint64(w & 0xFFFFFFFFFFFFFFFF)

    assert (planes_py == planes_cpp).all(), f"planes differ for {fen}"
    assert len(scalars_py) == len(scalars_cpp), "scalar count mismatch"
    assert np.allclose(scalars_py, scalars_cpp, atol=1e-5), \
        f"scalars differ: {scalars_py} vs {scalars_cpp}"
    assert mat_py == mat_cpp, f"material differs: {mat_py} vs {mat_cpp}"


def test_stm_disambiguation():
    """Same piece placement, opposite turn -> encodings MUST differ."""
    b1 = chess.Board("8/8/8/8/8/2k5/2p5/K7 w - - 0 1")
    b2 = chess.Board("8/8/8/8/8/2k5/2p5/K7 b - - 0 1")
    p1, s1 = encode_board(b1)
    p2, s2 = encode_board(b2)
    assert (p1 == p2).all(), "piece planes should match"
    assert not np.allclose(s1, s2), "scalars must disambiguate side to move"
    assert abs(s1[3] - s2[3]) == 1.0   # stm scalar flips
