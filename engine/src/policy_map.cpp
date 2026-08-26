#include "policy_map.h"
#include <cassert>

namespace chess {

int move_policy_index(const Move& m, Color mover) {
    const int df = file_of(m.to) - file_of(m.from);
    const int dr = rank_of(m.to) - rank_of(m.from);
    const int rel_dr = (mover == WHITE) ? dr : -dr;
    const int dist = std::max(std::abs(df), std::abs(dr));
    assert(dist >= 1 && dist <= 7);

    int plane = -1;
    if (m.promo >= 0 && m.promo != QUEEN) {
        int cap_dir = df < 0 ? 0 : (df > 0 ? 2 : 1);
        int p_idx = m.promo == KNIGHT ? 0 : (m.promo == BISHOP ? 1 : 2);
        plane = 64 + p_idx * 3 + cap_dir;
    } else if ((std::abs(df) == 1 && std::abs(dr) == 2) || (std::abs(df) == 2 && std::abs(dr) == 1)) {
        int k = knight_plane_index(df, rel_dr);
        if (k < 0) return -1;   // should never happen for legal knight moves
        plane = 56 + k;
    } else {
        plane = dir_index(df, rel_dr) * 7 + (dist - 1);
    }
    return plane * 64 + m.from;
}

}  // namespace chess
