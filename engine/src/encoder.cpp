#include "encoder.h"
#include <algorithm>

namespace chess {

int material_diff_stm(const Position& pos) {
    static constexpr int VAL[6] = {1, 3, 3, 5, 9, 0};
    int diff = 0;
    for (int pt = 0; pt < 6; ++pt) {
        diff += VAL[pt] * __builtin_popcountll(pos.bb[piece_idx(WHITE, static_cast<PieceType>(pt))]);
        diff -= VAL[pt] * __builtin_popcountll(pos.bb[piece_idx(BLACK, static_cast<PieceType>(pt))]);
    }
    return (pos.stm == WHITE) ? diff : -diff;
}

EncodedPosition encode_position(const Position& current, const std::vector<Position>& history) {
    return encode_position(current, history, {});
}

EncodedPosition encode_position(const Position& current,
                                const std::vector<Position>& game_hist,
                                const std::vector<Position>& walk) {
    EncodedPosition e;

    int rep_extra = 0;
    {
        uint64_t h = current.hash;
        int seen = -1;
        for (const Position& p : game_hist)
            if (p.hash == h) ++seen;
        for (const Position& p : walk)
            if (p.hash == h) ++seen;
        rep_extra = std::max(seen, 0);
    }

    auto at = [&](size_t idx_from_end) -> const Position* {
        if (idx_from_end < walk.size())
            return &walk[walk.size() - 1 - idx_from_end];
        const size_t k = idx_from_end - walk.size();
        if (k < game_hist.size())
            return &game_hist[game_hist.size() - 1 - k];
        return nullptr;
    };

    for (int a = 0; a < HIST_STEPS; ++a) {
        const int base = a * PLANES_PER_AGE;
        const Position* p = at(static_cast<size_t>(a));
        if (!p) continue;

        for (int pc = 0; pc < 12; ++pc) e.planes[base + pc] = p->bb[pc];
        e.planes[base + 12] = 0;
        e.planes[base + 13] = 0;
    }

    if (rep_extra >= 1) e.planes[12] = ~0ULL;
    if (rep_extra >= 2) e.planes[13] = ~0ULL;

    if (current.ep >= 0) e.planes[EP_PLANE] = 1ULL << current.ep;
    else e.planes[EP_PLANE] = 0;

    e.scalars[0] = std::min(current.halfmove / 100.0f, 1.0f);
    e.scalars[1] = std::min(current.fullmove / 200.0f, 1.0f);
    e.scalars[2] = std::min(rep_extra, 3) / 3.0f;
    e.scalars[3] = (current.stm == WHITE) ? 1.0f : 0.0f;
    e.scalars[4] = (current.castling & CASTLE_WK) ? 1.0f : 0.0f;
    e.scalars[5] = (current.castling & CASTLE_WQ) ? 1.0f : 0.0f;
    e.scalars[6] = (current.castling & CASTLE_BK) ? 1.0f : 0.0f;
    e.scalars[7] = (current.castling & CASTLE_BQ) ? 1.0f : 0.0f;
    e.scalars[8] = (current.ep >= 0) ? 1.0f : 0.0f;
    return e;
}

}  // namespace chess