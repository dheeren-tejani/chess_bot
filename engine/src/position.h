#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include "types.h"

namespace chess {

struct Position {
    // bb[color*6 + pieceType] bitboards; mailbox for O(1) piece lookup.
    std::array<uint64_t, 12> bb{};
    std::array<int8_t, 64> board{};   // -1 empty, else piece_idx
    uint8_t stm = WHITE;
    uint8_t castling = 0;
    int8_t ep = -1;                   // en passant target square or -1
    uint16_t halfmove = 0;
    uint16_t fullmove = 1;
    uint64_t hash = 0;

    uint64_t occ() const {
        return bb[0]|bb[1]|bb[2]|bb[3]|bb[4]|bb[5]|bb[6]|bb[7]|bb[8]|bb[9]|bb[10]|bb[11];
    }
    uint64_t occ_color(Color c) const {
        const uint64_t* b = &bb[static_cast<int>(c) * 6];
        return b[0]|b[1]|b[2]|b[3]|b[4]|b[5];
    }
    PieceType piece_at(int sq) const {
        int8_t p = board[sq];
        return p < 0 ? static_cast<PieceType>(-1) : static_cast<PieceType>(p % 6);
    }
    int king_sq(Color c) const { return __builtin_ctzll(bb[piece_idx(c, KING)]); }

    void set_from_fen(const std::string& fen);
    std::string to_fen() const;
};

// Attack/occupancy utilities (tables initialized once at startup).
namespace attacks {
void init();
bool bmi2_supported();

uint64_t pawn_attacks(Color c, int sq);
uint64_t knight(int sq);
uint64_t king(int sq);
uint64_t bishop(int sq, uint64_t occ);
uint64_t rook(int sq, uint64_t occ);
uint64_t queen(int sq, uint64_t occ);

// All attackers of `sq` by pieces of color `c`, given occupancy `occ`.
uint64_t attackers_to(const Position& pos, int sq, Color c, uint64_t occ);
inline bool square_attacked(const Position& pos, int sq, Color by, uint64_t occ) {
    return attackers_to(pos, sq, by, occ) != 0;
}
}  // namespace attacks

// Move application (copy-make). Returns the new position.
Position make_move(const Position& pos, const Move& m);

// Legal move generation.
void gen_legal_moves(const Position& pos, std::vector<Move>& out);

// Perft node count.
uint64_t perft(const Position& pos, int depth);

// Game-state helpers
enum class GameState { ONGOING, CHECKMATE, STALEMATE, DRAW_50MOVE, DRAW_MATERIAL };
GameState game_state(const Position& pos);   // does not consider repetition

// Insufficient material: KvK, KvK+minor, KvBvB same color (all immediate draws)
bool insufficient_material(const Position& pos);

}  // namespace chess
