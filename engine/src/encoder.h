#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include "position.h"

namespace chess {

// Input encoding - MUST match trainer/training/encoding.py exactly.
//
// Spatial planes (bit-packed, one uint64 word per plane, bit s = square s):
//   ages a = 0..7  (a = 0 is the CURRENT position; a>0 looks back `a` plies;
//                   missing history -> all-zero planes for that age)
//     base = a * 14
//     base+0..11 : piece planes, white P,N,B,R,Q,K then black P,N,B,R,Q,K
//     base+12    : this position occurred >= 1 extra time in game history
//     base+13    : this position occurred >= 2 extra times in game history
//   plane 112    : en-passant TARGET square (one-hot bit at that square)
//
// Scalar features (float):
//   [0] halfmove_clock / 100.0  (clamped to 1.0)
//   [1] fullmove / 200.0        (clamped to 1.0)
//   [2] min(repetition_count_of_current, 3) / 3.0
//   [3] side to move: 1.0 = WHITE, 0.0 = BLACK
//   [4] white kingside castle right  (1/0)
//   [5] white queenside castle right (1/0)
//   [6] black kingside castle right  (1/0)
//   [7] black queenside castle right (1/0)
//   [8] en-passant square exists     (1/0)
constexpr int HIST_STEPS = 8;
constexpr int PLANES_PER_AGE = 14;
constexpr int AGE_BLOCK_PLANES = HIST_STEPS * PLANES_PER_AGE;   // 112
constexpr int EP_PLANE = 112;
constexpr int NUM_PLANES = AGE_BLOCK_PLANES + 1;                // 113 spatial planes
constexpr int SCALAR_COUNT = 9;

struct EncodedPosition {
    std::array<uint64_t, NUM_PLANES> planes{};   // word i = bits of plane i over squares
    std::array<float, SCALAR_COUNT> scalars{};
};

// `history` includes the CURRENT position as its last element (oldest first).
EncodedPosition encode_position(const Position& current, const std::vector<Position>& history);

// Zero-copy variant: history and walk are indexed virtually (walk positions come
// after game positions; the leaf is walk.back()). Avoids concatenating vectors
// on every neural-network evaluation.
EncodedPosition encode_position(const Position& current,
                                const std::vector<Position>& game_hist,
                                const std::vector<Position>& walk);

int material_diff_stm(const Position& pos);   // P=1 N=3 B=3 R=5 Q=9 K=0

}  // namespace chess
