#pragma once
#include <cstdint>
#include <array>
#include <string>

namespace chess {

// Squares: a1 = 0, b1 = 1, ..., h8 = 63. rank = sq >> 3, file = sq & 7.
enum Color : int { WHITE = 0, BLACK = 1 };
enum PieceType : int { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5 };

constexpr int NUM_PIECE_TYPES = 6;
constexpr int NUM_PIECES = 12;          // color * 6 + type
constexpr int NUM_SQUARES = 64;

constexpr inline Color operator~(Color c) { return static_cast<Color>(c ^ 1); }
constexpr inline int piece_idx(Color c, PieceType pt) { return static_cast<int>(c) * 6 + static_cast<int>(pt); }

constexpr inline int rank_of(int sq) { return sq >> 3; }
constexpr inline int file_of(int sq) { return sq & 7; }
constexpr inline int make_square(int rank, int file) { return rank * 8 + file; }

// Castling right bits
enum CastleBits : int {
    CASTLE_WK = 1,
    CASTLE_WQ = 2,
    CASTLE_BK = 4,
    CASTLE_BQ = 8,
};

struct Move {
    int from = -1;
    int to = -1;
    int promo = -1;      // PieceType or -1
    int flags = 0;       // bit 0: en passant capture, bit 1: castling, bit 2: double push

    enum Flags : int { F_EP = 1, F_CASTLE = 2, F_DOUBLE = 4 };

    bool operator==(const Move& o) const { return from == o.from && to == o.to && promo == o.promo; }
};

// Canonical packed form used on disk/wire: bits [0:6) from, [6:12) to, [12:15) promo type (0 = none).
// Flags are NOT stored - they are derived from geometry by make_move.
inline uint16_t pack_move(const Move& m) {
    return static_cast<uint16_t>(m.from | (m.to << 6) | ((m.promo >= 0 ? m.promo : 0) << 12));
}
inline Move unpack_move(uint16_t v) {
    Move m;
    m.from = v & 63;
    m.to = (v >> 6) & 63;
    int p = (v >> 12) & 7;
    m.promo = p ? p : -1;
    return m;
}

// Direction deltas as (df, dr) pairs
struct Delta { int df, dr; };

// Queen directions indexed: 0=N,1=NE,2=E,3=SE,4=S,5=SW,6=W,7=NW
constexpr Delta QUEEN_DIRS[8] = {
    {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1},
};

// Knight move plane order (df, dr) - MUST match trainer/training.policy_map
constexpr Delta KNIGHT_DELTAS[8] = {
    {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
};

}  // namespace chess
