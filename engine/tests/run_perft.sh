#!/usr/bin/env bash
set -euo pipefail
cd ~/chess_bot/engine/build
E=./engine
fail=0
check() {  # fen depth expected
  got=$($E perft "$1" "$2" | awk '{print $4}')
  if [ "$got" = "$3" ]; then echo "OK   d$2 = $got"; else echo "FAIL d$2 got $got want $3 :: $1"; fail=1; fi
}
echo "--- startpos"
check 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' 3 8902
echo "--- kiwipete"
K='r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'
check "$K" 1 48; check "$K" 2 2039; check "$K" 3 97862; check "$K" 4 4085603
echo "--- pos3 (ep pins)"
P3='8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'
check "$P3" 1 14; check "$P3" 2 191; check "$P3" 3 2812; check "$P3" 4 43238; check "$P3" 5 674624
echo "--- pos4 (promotions)"
P4='r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1'
check "$P4" 1 6; check "$P4" 2 264; check "$P4" 3 9467; check "$P4" 4 422333
echo "--- pos4 mirrored"
P4B='r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1'
check "$P4B" 1 6; check "$P4B" 2 264; check "$P4B" 3 9467; check "$P4B" 4 422333
echo "--- pos5"
P5='rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8'
check "$P5" 1 44; check "$P5" 2 1486; check "$P5" 3 62379; check "$P5" 4 2103487
echo "--- pos6 (values verified against python-chess)"
P6='r4rk1/1pp1qppp/p1np1n2/2b1p1b1/2B1P1B1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10'
check "$P6" 1 46; check "$P6" 2 2060; check "$P6" 3 88933
exit $fail
