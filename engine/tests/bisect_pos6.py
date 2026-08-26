#!/usr/bin/env python3
"""Compare engine perft-split against python-chess ground truth, per root move."""
import subprocess, sys
import chess

FEN = sys.argv[1] if len(sys.argv) > 1 else 'r4rk1/1pp1qppp/p1np1n2/2b1p1b1/2B1P1B1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10'
DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ENGINE = "/home/dheer/chess_bot/engine/build/engine"

out = subprocess.run([ENGINE, "perft-split", FEN, str(DEPTH)], capture_output=True, text=True).stdout
engine_counts = {}
for line in out.strip().splitlines():
    uci, n = line.split()
    engine_counts[uci] = int(n)

board = chess.Board(FEN)
ok = True
total_ref = 0
for mv in board.legal_moves:
    board.push(mv)
    ref = count = 0
    # perft depth-1 via legal move list
    stack = [(board, DEPTH - 1)]
    def perft(b, d):
        if d == 0:
            return 1
        if d == 1:
            return b.legal_moves.count()
        n = 0
        for m in b.legal_moves:
            b.push(m)
            n += perft(b, d - 1)
            b.pop()
        return n
    ref = perft(board, DEPTH - 1)
    total_ref += ref
    board.pop()
    uci = mv.uci()
    eng = engine_counts.get(uci)
    mark = ""
    if eng != ref:
        ok = False
        mark = f"   <<< MISMATCH engine={eng}"
    print(f"{uci:8s} ref={ref:6d}{mark}")
print(f"total ref={total_ref} engine={sum(engine_counts.values())}")
sys.exit(0 if ok else 1)
