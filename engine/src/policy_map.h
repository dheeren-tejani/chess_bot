#pragma once
#include "types.h"

namespace chess {

// AlphaZero-style policy target layout: 73 planes x 64 squares = 4672 logits.
// Index = plane * 64 + from_square.
//
// Planes (relative to the mover; vertical axis flipped for BLACK):
//   [0 ..55]  queen-like: dir*7 + (dist-1);  dir: 0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW
//             (N = toward the opponent)
//   [56..63]  knight moves, ordered per KNIGHT_DELTAS in types.h
//   [64..72]  underpromotions: pieceIdx*3 + capDir
//             pieceIdx: 0=N,1=B,2=R ; capDir: 0=file-1, 1=straight, 2=file+1
// Promotions to QUEEN ride the normal queen-like planes.
constexpr int POLICY_PLANES = 73;
constexpr int POLICY_ACTIONS = POLICY_PLANES * 64;

inline int dir_index(int df, int rel_dr) {
    // (df, rel_dr) -> 0..7 ; assumes unit step
    if (rel_dr > 0) return df == 0 ? 0 : (df > 0 ? 1 : 7);
    if (rel_dr < 0) return df == 0 ? 4 : (df > 0 ? 3 : 5);
    return df > 0 ? 2 : 6;
}

inline int knight_plane_index(int df, int rel_dr) {
    for (int i = 0; i < 8; ++i)
        if (KNIGHT_DELTAS[i].df == df && KNIGHT_DELTAS[i].dr == rel_dr) return i;
    return -1;
}

// `mover` = side making the move (needed for vertical mirroring).
int move_policy_index(const Move& m, Color mover);

// Inverse: enumerate every action index reachable from a position's legal moves.
// (Python mirrors this logic exactly; parity is covered by tests.)

}  // namespace chess
