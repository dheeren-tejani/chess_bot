#include "position.h"
#include "zobrist.h"
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <cctype>

namespace chess {
namespace attacks {

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace {
std::array<std::array<uint64_t, 64>, 2> PAWN_ATT;
std::array<uint64_t, 64> KNIGHT_T, KING_T;
std::array<uint64_t, 64> BISHOP_MASK, ROOK_MASK;
std::array<std::array<uint64_t, 512>, 64> BISHOP_TABLE;
std::array<std::array<uint64_t, 4096>, 64> ROOK_TABLE;
bool BMI2 = false;
bool INITED = false;

uint64_t slide(int sq, int df, int dr, uint64_t occ) {
    uint64_t a = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (;;) {
        f += df; r += dr;
        if (f < 0 || f > 7 || r < 0 || r > 7) break;
        int s = make_square(r, f);
        a |= 1ULL << s;
        if (occ & (1ULL << s)) break;
    }
    return a;
}

uint64_t slow_bishop(int sq, uint64_t occ) {
    uint64_t a = 0;
    for (auto d : {Delta{1,1}, Delta{1,-1}, Delta{-1,1}, Delta{-1,-1}}) a |= slide(sq, d.df, d.dr, occ);
    return a & ~(1ULL << sq);
}
uint64_t slow_rook(int sq, uint64_t occ) {
    uint64_t a = 0;
    for (auto d : {Delta{0,1}, Delta{0,-1}, Delta{1,0}, Delta{-1,0}}) a |= slide(sq, d.df, d.dr, occ);
    return a & ~(1ULL << sq);
}
}  // namespace

void init() {
    if (INITED) return;
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    BMI2 = __builtin_cpu_supports("bmi2");
#endif
    for (int sq = 0; sq < 64; ++sq) {
        int f = file_of(sq), r = rank_of(sq);
        uint64_t& wp = PAWN_ATT[WHITE][sq];
        uint64_t& bp = PAWN_ATT[BLACK][sq];
        wp = bp = KNIGHT_T[sq] = KING_T[sq] = 0;
        auto add = [&](uint64_t& m, int ff, int rr) {
            if (ff >= 0 && ff <= 7 && rr >= 0 && rr <= 7) m |= 1ULL << make_square(rr, ff);
        };
        add(wp, f - 1, r + 1); add(wp, f + 1, r + 1);
        add(bp, f - 1, r - 1); add(bp, f + 1, r - 1);
        for (auto d : KNIGHT_DELTAS) add(KNIGHT_T[sq], f + d.df, r + d.dr);
        for (auto d : QUEEN_DIRS)
            if (abs(d.df) <= 1 && abs(d.dr) <= 1) add(KING_T[sq], f + d.df, r + d.dr);

        // Relevant occupancy masks: squares strictly inside the board along each ray
        uint64_t bm = 0, rm = 0;
        for (int i = 1; i <= 6; ++i) {
            auto inner = [&](int df, int dr) {
                int nf = f + df * i, nr = r + dr * i;
                return nf >= 1 && nf <= 6 && nr >= 1 && nr <= 6;
            };
            auto tryadd_b = [&](int df, int dr) {
                if (inner(df, dr)) bm |= 1ULL << make_square(r + dr * i, f + df * i);
            };
            tryadd_b(1, 1); tryadd_b(1, -1); tryadd_b(-1, 1); tryadd_b(-1, -1);
        }
        for (int ff = 1; ff < 7; ++ff) rm |= 1ULL << make_square(r, ff);
        for (int rr = 1; rr < 7; ++rr) rm |= 1ULL << make_square(rr, f);
        rm &= ~(1ULL << sq);
        bm &= ~(1ULL << sq);
        BISHOP_MASK[sq] = bm;
        ROOK_MASK[sq] = rm;
        uint64_t bsub = bm;
        do {
            BISHOP_TABLE[sq][_pext_u64(bsub, bm)] = slow_bishop(sq, bsub);
            bsub = (bsub - 1) & bm;
        } while (bsub != bm);
        uint64_t rsub = rm;
        do {
            ROOK_TABLE[sq][_pext_u64(rsub, rm)] = slow_rook(sq, rsub);
            rsub = (rsub - 1) & rm;
        } while (rsub != rm);
    }
    INITED = true;
}

bool bmi2_supported() { return BMI2; }

uint64_t pawn_attacks(Color c, int sq) { return PAWN_ATT[c][sq]; }
uint64_t knight(int sq) { return KNIGHT_T[sq]; }
uint64_t king(int sq) { return KING_T[sq]; }
uint64_t bishop(int sq, uint64_t occ) {
    if (BMI2) return BISHOP_TABLE[sq][_pext_u64(occ & BISHOP_MASK[sq], BISHOP_MASK[sq])];
    return slow_bishop(sq, occ);  // portable fallback
}
uint64_t rook(int sq, uint64_t occ) {
    if (BMI2) return ROOK_TABLE[sq][_pext_u64(occ & ROOK_MASK[sq], ROOK_MASK[sq])];
    return slow_rook(sq, occ);
}
uint64_t queen(int sq, uint64_t occ) { return bishop(sq, occ) | rook(sq, occ); }

uint64_t attackers_to(const Position& pos, int sq, Color c, uint64_t occ) {
    const uint64_t* b = &pos.bb[static_cast<int>(c) * 6];
    uint64_t a = 0;
    a |= pawn_attacks(~c, sq) & b[PAWN];
    a |= knight(sq) & b[KNIGHT];
    a |= king(sq) & b[KING];
    a |= bishop(sq, occ) & (b[BISHOP] | b[QUEEN]);
    a |= rook(sq, occ) & (b[ROOK] | b[QUEEN]);
    return a;
}

}  // namespace attacks

// ---------------------------------------------------------------------------
// FEN
// ---------------------------------------------------------------------------
void Position::set_from_fen(const std::string& fen) {
    *this = Position{};
    board.fill(-1);   // empty squares must be -1, not 0 (= white pawn)!
    const Zobrist& Z = Zobrist::instance();
    std::istringstream ss(fen);
    std::string board_s, stm_s, castle_s, ep_s;
    int hm = 0, fm = 1;
    ss >> board_s >> stm_s >> castle_s >> ep_s >> hm >> fm;
    if (board_s.empty()) throw std::invalid_argument("bad fen");

    int r = 7, f = 0;
    for (char ch : board_s) {
        if (ch == '/') { --r; f = 0; continue; }
        if (isdigit((unsigned char)ch)) { f += ch - '0'; continue; }
        Color c = isupper((unsigned char)ch) ? WHITE : BLACK;
        PieceType pt;
        switch (tolower((unsigned char)ch)) {
            case 'p': pt = PAWN; break;   case 'n': pt = KNIGHT; break;
            case 'b': pt = BISHOP; break; case 'r': pt = ROOK; break;
            case 'q': pt = QUEEN; break;  case 'k': pt = KING; break;
            default: throw std::invalid_argument("bad fen piece");
        }
        int sq = make_square(r, f);
        bb[piece_idx(c, pt)] |= 1ULL << sq;
        board[sq] = static_cast<int8_t>(piece_idx(c, pt));
        ++f;
    }
    stm = (stm_s == "w") ? WHITE : BLACK;
    castling = 0;
    for (char ch : castle_s) {
        if (ch == 'K') castling |= CASTLE_WK;
        else if (ch == 'Q') castling |= CASTLE_WQ;
        else if (ch == 'k') castling |= CASTLE_BK;
        else if (ch == 'q') castling |= CASTLE_BQ;
    }
    ep = -1;
    if (!ep_s.empty() && ep_s != "-") {
        int f = ep_s[0] - 'a', r = ep_s[1] - '1';
        ep = static_cast<int8_t>(make_square(r, f));
    }
    halfmove = static_cast<uint16_t>(hm);
    fullmove = static_cast<uint16_t>(fm);

    hash = 0;
    for (int s = 0; s < 64; ++s)
        if (board[s] >= 0) hash ^= Z.psq[board[s]][s];
    hash ^= Z.castle[castling];
    if (ep >= 0) hash ^= Z.epfile[file_of(ep)];
    if (stm == BLACK) hash ^= Z.side;
}

std::string Position::to_fen() const {
    std::ostringstream os;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int p = board[make_square(r, f)];
            if (p < 0) { ++empty; continue; }
            if (empty) { os << empty; empty = 0; }
            char ch = "pnbrqk"[p % 6];
            os << (p < 6 ? static_cast<char>(toupper(ch)) : ch);
        }
        if (empty) os << empty;
        if (r) os << '/';
    }
    os << (stm == WHITE ? " w " : " b ");
    if (!castling) os << '-';
    else {
        if (castling & CASTLE_WK) os << 'K';
        if (castling & CASTLE_WQ) os << 'Q';
        if (castling & CASTLE_BK) os << 'k';
        if (castling & CASTLE_BQ) os << 'q';
    }
    os << ' ';
    if (ep < 0) os << '-';
    else os << static_cast<char>('a' + file_of(ep)) << static_cast<char>('1' + rank_of(ep));
    os << ' ' << halfmove << ' ' << fullmove;
    return os.str();
}

// ---------------------------------------------------------------------------
// make_move
// ---------------------------------------------------------------------------
namespace {
constexpr std::array<uint8_t, 64> castle_masks() {
    std::array<uint8_t, 64> m{};
    m.fill(15);
    m[make_square(0, 4)] = 15 & ~(CASTLE_WK | CASTLE_WQ);   // e1
    m[make_square(0, 7)] = 15 & ~CASTLE_WK;                 // h1
    m[make_square(0, 0)] = 15 & ~CASTLE_WQ;                 // a1
    m[make_square(7, 4)] = 15 & ~(CASTLE_BK | CASTLE_BQ);   // e8
    m[make_square(7, 7)] = 15 & ~CASTLE_BK;                 // h8
    m[make_square(7, 0)] = 15 & ~CASTLE_BQ;                 // a8
    return m;
}
const auto CASTLE_MASK = castle_masks();
}  // namespace

Position make_move(const Position& pos, const Move& m) {
    const Zobrist& Z = Zobrist::instance();
    Position n = pos;

    const int us = pos.stm, them = pos.stm ^ 1;
    const int pc = pos.board[m.from];
    assert(pc >= 0);

    // Derive move nature from geometry (never trust external flag bits).
    const bool is_pawn = (pc % 6) == PAWN;
    const bool is_king = (pc % 6) == KING;
    const int df = file_of(m.to) - file_of(m.from);
    const int dr = rank_of(m.to) - rank_of(m.from);
    bool is_castle = is_king && (df == 2 || df == -2);
    bool is_ep = false, is_double = false;
    if (is_pawn && pos.ep >= 0 && m.to == pos.ep && df != 0) is_ep = true;
    if (is_pawn && dr == 2 * (us == WHITE ? 1 : -1)) {
        // double push only if origin on start rank
        if ((us == WHITE && rank_of(m.from) == 1) || (us == BLACK && rank_of(m.from) == 6))
            is_double = true;
    }

    // remove old ep / castle keys from hash
    n.hash ^= Z.castle[n.castling];
    if (n.ep >= 0) n.hash ^= Z.epfile[file_of(n.ep)];
    n.ep = -1;

    n.halfmove = static_cast<uint16_t>(n.halfmove + 1);

    // capture?
    int captured = n.board[m.to];
    if (captured >= 0) {
        n.bb[captured] &= ~(1ULL << m.to);
        n.hash ^= Z.psq[captured][m.to];
        n.board[m.to] = -1;
        n.halfmove = 0;
    }

    // en passant capture: remove the captured pawn
    if (is_ep) {
        int cap_sq = (us == WHITE) ? m.to - 8 : m.to + 8;
        int cap_pc = piece_idx(static_cast<Color>(them), PAWN);
        n.bb[cap_pc] &= ~(1ULL << cap_sq);
        n.hash ^= Z.psq[cap_pc][cap_sq];
        n.board[cap_sq] = -1;
        n.halfmove = 0;
    }

    // move the piece
    n.bb[pc] &= ~(1ULL << m.from);
    n.hash ^= Z.psq[pc][m.from];

    int placed = pc;
    if (m.promo >= 0) placed = piece_idx(static_cast<Color>(us), static_cast<PieceType>(m.promo));

    n.bb[placed] |= 1ULL << m.to;
    n.board[m.to] = static_cast<int8_t>(placed);
    n.hash ^= Z.psq[placed][m.to];

    if (is_pawn) n.halfmove = 0;

    // castling rook move
    if (is_castle) {
        bool kingside = file_of(m.to) == 6;
        int rf = (us == WHITE) ? (kingside ? make_square(0, 7) : make_square(0, 0))
                               : (kingside ? make_square(7, 7) : make_square(7, 0));
        int rt = (us == WHITE) ? (kingside ? make_square(0, 5) : make_square(0, 3))
                               : (kingside ? make_square(7, 5) : make_square(7, 3));
        int rk = piece_idx(static_cast<Color>(us), ROOK);
        n.bb[rk] &= ~(1ULL << rf);
        n.bb[rk] |= 1ULL << rt;
        n.board[rf] = -1;
        n.board[rt] = static_cast<int8_t>(rk);
        n.hash ^= Z.psq[rk][rf] ^ Z.psq[rk][rt];
    }

    // double push -> set ep square
    if (is_double) {
        n.ep = static_cast<int8_t>((us == WHITE) ? m.to - 8 : m.to + 8);
    }

    n.castling &= CASTLE_MASK[m.from] & CASTLE_MASK[m.to];

    // re-add new keys
    n.hash ^= Z.castle[n.castling];
    if (n.ep >= 0) n.hash ^= Z.epfile[file_of(n.ep)];

    n.stm = static_cast<uint8_t>(them);
    n.hash ^= Z.side;
    if (us == BLACK) n.fullmove = static_cast<uint16_t>(n.fullmove + 1);
    return n;
}

// ---------------------------------------------------------------------------
// Move generation
// ---------------------------------------------------------------------------
namespace {

template <Color US>
void gen_pseudo(const Position& pos, std::vector<Move>& out) {
    constexpr Color THEM = ~US;
    const uint64_t* b = &pos.bb[static_cast<int>(US) * 6];
    const uint64_t own = pos.occ_color(US);
    const uint64_t opp = pos.occ_color(THEM);
    const uint64_t all = own | opp;
    const uint64_t targets = ~own;

    auto push = [&](int from, int to, int promo = -1, int flags = 0) {
        out.push_back(Move{from, to, promo, flags});
    };

    // Pawns
    uint64_t pawns = b[PAWN];
    while (pawns) {
        int from = __builtin_ctzll(pawns);
        pawns &= pawns - 1;
        int r = rank_of(from), f = file_of(from);
        if constexpr (US == WHITE) {
            int to = from + 8;
            if (!(all & (1ULL << to))) {
                if (r == 6) { push(from, to, QUEEN); push(from, to, ROOK); push(from, to, BISHOP); push(from, to, KNIGHT); }
                else {
                    push(from, to);
                    if (r == 1 && !(all & (1ULL << (from + 16)))) push(from, from + 16, -1, Move::F_DOUBLE);
                }
            }
            uint64_t caps = attacks::pawn_attacks(WHITE, from) & opp;
            while (caps) {
                int to = __builtin_ctzll(caps); caps &= caps - 1;
                if (r == 6) { push(from, to, QUEEN); push(from, to, ROOK); push(from, to, BISHOP); push(from, to, KNIGHT); }
                else push(from, to);
            }
            if (pos.ep >= 0 && (attacks::pawn_attacks(WHITE, from) & (1ULL << pos.ep)))
                push(from, pos.ep, -1, Move::F_EP);
        } else {
            int to = from - 8;
            if (!(all & (1ULL << to))) {
                if (r == 1) { push(from, to, QUEEN); push(from, to, ROOK); push(from, to, BISHOP); push(from, to, KNIGHT); }
                else {
                    push(from, to);
                    if (r == 6 && !(all & (1ULL << (from - 16)))) push(from, from - 16, -1, Move::F_DOUBLE);
                }
            }
            uint64_t caps = attacks::pawn_attacks(BLACK, from) & opp;
            while (caps) {
                int to = __builtin_ctzll(caps); caps &= caps - 1;
                if (r == 1) { push(from, to, QUEEN); push(from, to, ROOK); push(from, to, BISHOP); push(from, to, KNIGHT); }
                else push(from, to);
            }
            if (pos.ep >= 0 && (attacks::pawn_attacks(BLACK, from) & (1ULL << pos.ep)))
                push(from, pos.ep, -1, Move::F_EP);
        }
        (void)f;
    }

    // Knights
    uint64_t kn = b[KNIGHT];
    while (kn) {
        int from = __builtin_ctzll(kn); kn &= kn - 1;
        uint64_t t = attacks::knight(from) & targets;
        while (t) { int to = __builtin_ctzll(t); t &= t - 1; push(from, to); }
    }

    // Bishops / Rooks / Queens
    uint64_t bi = b[BISHOP];
    while (bi) {
        int from = __builtin_ctzll(bi); bi &= bi - 1;
        uint64_t t = attacks::bishop(from, all) & targets;
        while (t) { int to = __builtin_ctzll(t); t &= t - 1; push(from, to); }
    }
    uint64_t rk = b[ROOK];
    while (rk) {
        int from = __builtin_ctzll(rk); rk &= rk - 1;
        uint64_t t = attacks::rook(from, all) & targets;
        while (t) { int to = __builtin_ctzll(t); t &= t - 1; push(from, to); }
    }
    uint64_t qn = b[QUEEN];
    while (qn) {
        int from = __builtin_ctzll(qn); qn &= qn - 1;
        uint64_t t = attacks::queen(from, all) & targets;
        while (t) { int to = __builtin_ctzll(t); t &= t - 1; push(from, to); }
    }

    // King
    {
        int from = pos.king_sq(US);
        uint64_t t = attacks::king(from) & targets;
        while (t) { int to = __builtin_ctzll(t); t &= t - 1; push(from, to); }

        // Castling
        if constexpr (US == WHITE) {
            if ((pos.castling & CASTLE_WK) && !(all & 0x60ULL)               // f1,g1
                && !attacks::square_attacked(pos, 4, BLACK, all)
                && !attacks::square_attacked(pos, 5, BLACK, all))
                push(from, 6, -1, Move::F_CASTLE);
            if ((pos.castling & CASTLE_WQ) && !(all & 0xEULL)                // b1,c1,d1
                && !attacks::square_attacked(pos, 4, BLACK, all)
                && !attacks::square_attacked(pos, 3, BLACK, all))
                push(from, 2, -1, Move::F_CASTLE);
        } else {
            if ((pos.castling & CASTLE_BK) && !(all & 0x6000000000000000ULL) // f8,g8
                && !attacks::square_attacked(pos, 60, WHITE, all)
                && !attacks::square_attacked(pos, 61, WHITE, all))
                push(from, 62, -1, Move::F_CASTLE);
            if ((pos.castling & CASTLE_BQ) && !(all & 0xE00000000000000ULL)  // b8,c8,d8
                && !attacks::square_attacked(pos, 60, WHITE, all)
                && !attacks::square_attacked(pos, 59, WHITE, all))
                push(from, 58, -1, Move::F_CASTLE);
        }
    }
}

}  // namespace

void gen_legal_moves(const Position& pos, std::vector<Move>& out) {
    out.clear();
    std::vector<Move> pseudo;
    pseudo.reserve(48);
    if (pos.stm == WHITE) gen_pseudo<WHITE>(pos, pseudo);
    else gen_pseudo<BLACK>(pos, pseudo);

    const int us = pos.stm;
    out.reserve(pseudo.size());
    for (const Move& m : pseudo) {
        Position np = make_move(pos, m);
        if (!attacks::square_attacked(np, np.king_sq(static_cast<Color>(us)),
                                      static_cast<Color>(us ^ 1), np.occ()))
            out.push_back(m);
    }
}

uint64_t perft(const Position& pos, int depth) {
    if (depth == 0) return 1;
    std::vector<Move> moves;
    gen_legal_moves(pos, moves);
    if (depth == 1) return moves.size();
    uint64_t nodes = 0;
    for (const Move& m : moves)
        nodes += perft(make_move(pos, m), depth - 1);
    return nodes;
}

// ---------------------------------------------------------------------------
// Game state
// ---------------------------------------------------------------------------
bool insufficient_material(const Position& pos) {
    uint64_t all = pos.occ();
    if (__builtin_popcountll(all) == 2) return true;  // KvK
    if (__builtin_popcountll(all) == 3) {
        int extra = __builtin_ctzll(all & ~(pos.bb[piece_idx(WHITE, KING)] | pos.bb[piece_idx(BLACK, KING)]));
        int8_t p = pos.board[extra];
        if (p >= 0 && (p % 6) == KNIGHT) return true;
        if (p >= 0 && (p % 6) == BISHOP) return true;
    }
    // KvB vs KvB same color bishops
    uint64_t minors = (pos.bb[piece_idx(WHITE, BISHOP)] | pos.bb[piece_idx(BLACK, BISHOP)]);
    bool only_bishops = (all & ~(minors | pos.bb[piece_idx(WHITE, KING)] | pos.bb[piece_idx(BLACK, KING)])) == 0
                        && __builtin_popcountll(minors) == __builtin_popcountll(all) - 2;
    if (only_bishops) {
        int color = __builtin_ctzll(minors) & 1;
        uint64_t tmp = minors;
        bool same = true;
        while (tmp) { int s = __builtin_ctzll(tmp); tmp &= tmp - 1; if ((s & 1) != color) same = false; }
        if (same) return true;
    }
    return false;
}

GameState game_state(const Position& pos) {
    std::vector<Move> moves;
    gen_legal_moves(pos, moves);
    if (moves.empty()) {
        if (attacks::square_attacked(pos, pos.king_sq(static_cast<Color>(pos.stm)),
                                     static_cast<Color>(pos.stm ^ 1), pos.occ()))
            return GameState::CHECKMATE;
        return GameState::STALEMATE;
    }
    if (pos.halfmove >= 100) return GameState::DRAW_50MOVE;
    if (insufficient_material(pos)) return GameState::DRAW_MATERIAL;
    return GameState::ONGOING;
}

}  // namespace chess
