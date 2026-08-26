#include "mcts.h"
#include <algorithm>
#include <cmath>
#include <random>

namespace chess {

Search::Search(const MCTSConfig& cfg, std::vector<Position>&& game_history,
               std::mt19937_64* rng)
    : cfg_(cfg), hist_(std::make_unique<std::vector<Position>>(std::move(game_history))),
      rng_(rng) {
    root_ = tree_.new_node();
}

uint32_t Search::node_visits(const Tree::Node& n, const Tree::Edge* es) {
    uint32_t v = 0;
    for (int i = 0; i < n.num; ++i) v += es[i].visits;
    return v;
}

float Search::cpuct(float parent_visits) const {
    return cfg_.cpuct_init +
           std::log((parent_visits + cfg_.cpuct_base + 1.0f) / cfg_.cpuct_base);
}

int Search::root_visits() const {
    return static_cast<int>(node_visits(tree_.node(root_), tree_.edges(root_)));
}

float Search::root_q() const {
    const Tree::Node& n = tree_.node(root_);
    const Tree::Edge* es = tree_.edges(root_);
    double s = 0;
    uint32_t v = 0;
    for (int i = 0; i < n.num; ++i) { s += es[i].value_sum; v += es[i].visits; }
    return v ? static_cast<float>(s / v) : 0.0f;
}

int Search::select_child(int node, std::vector<PathEntry>& path) {
    const Tree::Node& n = tree_.node(node);
    Tree::Edge* es = tree_.edges(node);

    uint32_t total_v = 0;
    double total_s = 0.0;
    for (int i = 0; i < n.num; ++i) { total_v += es[i].visits; total_s += es[i].value_sum; }

    float parent_q = total_v ? static_cast<float>(total_s / total_v) : 0.0f;
    float fpu = parent_q - cfg_.fpu_reduction;
    float sqrt_n = std::sqrt(static_cast<float>(std::max<uint32_t>(total_v, 1u)));
    float c = cpuct(static_cast<float>(total_v));

    int best = -1;
    float best_score = -1e30f;
    for (int i = 0; i < n.num; ++i) {
        const Tree::Edge& e = es[i];
        float q = e.visits ? e.value_sum / static_cast<float>(e.visits) : fpu;
        float u = c * (static_cast<float>(e.prior) * (1.0f / 65535.0f)) * sqrt_n /
                  (1.0f + static_cast<float>(e.visits));
        float score = q + u;
        if (score > best_score) { best_score = score; best = i; }
    }
    path.push_back({node, best});
    return best;
}

// Virtual-loss contract: descent subtracts VL from each traversed edge once.
// backprop adds (contrib + VL) per traversed edge, compensating exactly.
void Search::backprop(const std::vector<PathEntry>& path, float value_from_leaf_stm) {
    float contrib = -value_from_leaf_stm;   // deepest edge: flip to parent perspective
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Tree::Edge& e = tree_.edges(it->node)[it->edge];
        e.visits += 1;
        e.value_sum += contrib + cfg_.virtual_loss;
        contrib = -contrib;
    }
}

void Search::make_task(const Position& leaf, const std::vector<Position>& walk,
                       const std::vector<Move>& legal, int node_idx,
                       const std::vector<PathEntry>& path,
                       std::vector<int>& out_task_slots) {
    EvalTask t;
    t.moves.reserve(legal.size());
    t.priors.assign(legal.size(), 0.0f);
    for (const Move& m : legal) t.moves.push_back(pack_move(m));
    t.stm = leaf.stm;
    t.path.reserve(path.size());
    for (const PathEntry& pe : path) t.path.emplace_back(pe.node, pe.edge);

    // zero-copy encoding over (game history, search walk)
    t.enc = encode_position(leaf, *hist_, walk);

    int slot = static_cast<int>(tasks_.size());
    t.node = node_idx;
    tasks_.push_back(std::move(t));
    tree_.node(node_idx).state = Tree::ST_INFLIGHT;
    tree_.node(node_idx).task_slot = slot;
    ++live_tasks_;
    out_task_slots.push_back(slot);
}

void Search::compensate_vl(const std::vector<PathEntry>& path) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Tree::Edge& e = tree_.edges(it->node)[it->edge];
        e.value_sum += cfg_.virtual_loss;
    }
}

void Search::descend_once(std::vector<int>& out_task_slots) {
    std::vector<PathEntry> path;
    std::vector<Position> walk;
    int node = root_;

    auto cur = [&]() -> const Position& { return walk.empty() ? hist_->back() : walk.back(); };

    for (;;) {
        Tree::Node& n = tree_.node(node);

        if (n.state == Tree::ST_TERMINAL) {
            backprop(path, n.term_value);
            return;
        }
        if (n.state == Tree::ST_INFLIGHT) {
            // Aborted descent: undo VL, do NOT count visits or values.
            compensate_vl(path);
            return;
        }
        if (n.state == Tree::ST_UNEXPANDED) {
            const Position& leaf = cur();
            thread_local std::vector<Move> legal;   // reused scratch, no realloc churn
            legal.clear();
            gen_legal_moves(leaf, legal);
            if (legal.empty()) {
                bool check = attacks::square_attacked(
                    leaf, leaf.king_sq(static_cast<Color>(leaf.stm)),
                    static_cast<Color>(leaf.stm ^ 1), leaf.occ());
                n.state = Tree::ST_TERMINAL;
                n.term_value = check ? -1.0f : 0.0f;   // stm is mated : stalemate
                backprop(path, n.term_value);
                return;
            }
            make_task(leaf, walk, legal, node, path, out_task_slots);
            // no backprop here: complete_task will count this playout
            return;
        }

        // EXPANDED: select and descend one level
        int ei = select_child(node, path);
        Tree::Edge& e = tree_.edges(node)[ei];
        e.value_sum -= cfg_.virtual_loss;

        if (e.child < 0) e.child = tree_.new_node();

        Position next = make_move(cur(), unpack_move(e.move));

        if (cfg_.use_twofold_draw && next.hash == cur().hash) {
            // immediate repetition of the current position
            Tree::Node& cn = tree_.node(e.child);
            cn.state = Tree::ST_TERMINAL;
            cn.term_value = 0.0f;
            backprop(path, 0.0f);   // includes this edge
            return;
        }
        if (cfg_.use_twofold_draw) {
            int seen = 0;
            for (const Position& p : *hist_)
                if (p.hash == next.hash) ++seen;
            for (const Position& p : walk)
                if (p.hash == next.hash) ++seen;
            if (seen >= 2) {
                Tree::Node& cn = tree_.node(e.child);
                cn.state = Tree::ST_TERMINAL;
                cn.term_value = 0.0f;
                backprop(path, 0.0f);
                return;
            }
        }
        if (next.halfmove >= 100) {
            Tree::Node& cn = tree_.node(e.child);
            cn.state = Tree::ST_TERMINAL;
            cn.term_value = 0.0f;
            backprop(path, 0.0f);
            return;
        }

        walk.push_back(std::move(next));
        node = e.child;
    }
}

void Search::gather_round(int max_leaves, std::vector<int>& out_task_slots) {
    for (int i = 0; i < max_leaves; ++i)
        descend_once(out_task_slots);
}

void Search::maybe_noise_root() {
    Tree::Node& n = tree_.node(root_);
    if (n.noised || !cfg_.root_dirichlet) return;
    if (n.num <= 0) return;

    std::gamma_distribution<float> gamma(cfg_.dirichlet_alpha, 1.0f);
    std::vector<float> noise(n.num);
    float sum = 0;
    for (auto& x : noise) { x = gamma(*rng_); sum += x; }
    if (sum <= 0) return;
    for (auto& x : noise) x /= sum;

    Tree::Edge* es = tree_.edges(root_);
    for (int i = 0; i < n.num; ++i) {
        float p = (1.0f - cfg_.dirichlet_eps) * (es[i].prior / 65535.0f) +
                  cfg_.dirichlet_eps * noise[i];
        es[i].prior = static_cast<uint16_t>(
            std::min(65535.0f, std::max(0.0f, p * 65535.0f)));
    }
    n.noised = 1;
}

void Search::complete_task(int slot) {
    EvalTask& t = tasks_[slot];
    Tree::Node& n = tree_.node(t.node);
    if (n.state != Tree::ST_INFLIGHT) return;   // defensive
    --live_tasks_;

    // materialize children with evaluated priors
    std::vector<Tree::Edge> es;
    es.reserve(t.moves.size());
    for (size_t i = 0; i < t.moves.size(); ++i) {
        Tree::Edge e;
        e.move = t.moves[i];
        Move m = unpack_move(t.moves[i]);
        e.policy_idx = static_cast<int16_t>(move_policy_index(m, static_cast<Color>(t.stm)));
        e.prior = static_cast<uint16_t>(
            std::min(65535.0f, std::max(0.0f, t.priors[i] * 65535.0f)));
        e.child = -1;
        es.push_back(e);
    }
    tree_.attach_edges(t.node, std::move(es));
    n.state = Tree::ST_EXPANDED;
    n.task_slot = -1;

    if (t.node == root_) maybe_noise_root();

    // fold this playout's value into the tree along its recorded path
    std::vector<PathEntry> path;
    path.reserve(t.path.size());
    for (const auto& [nd, ed] : t.path) path.push_back({nd, ed});
    backprop(path, t.value);
}

Search::Choice Search::pick_move(std::mt19937_64& rng, int ply_in_game) {
    const Tree::Node& n = tree_.node(root_);
    const Tree::Edge* es = tree_.edges(root_);
    Choice ch;
    if (n.num <= 0) return ch;

    std::vector<uint32_t> visits(n.num);
    uint32_t total = 0;
    for (int i = 0; i < n.num; ++i) { visits[i] = es[i].visits; total += visits[i]; }

    int pick = -1;

    // 1) opening plies: sample the (Dirichlet-noised) prior distribution.
    //    Priors are already noise-mixed at the root (maybe_noise_root runs on
    //    every new root), so this inherits exploration for free.
    if (ply_in_game < cfg_.prior_plies) {
        uint32_t ptot = 0;
        for (int i = 0; i < n.num; ++i) ptot += es[i].prior;
        if (ptot > 0) {
            std::uniform_int_distribution<uint32_t> dist(1, ptot);
            uint32_t roll = dist(rng);
            for (int i = 0; i < n.num; ++i) {
                if (roll <= es[i].prior) { pick = i; break; }
                roll -= es[i].prior;
            }
            if (pick < 0) pick = n.num - 1;
        }
    }

    // 2) early middle-game: sample by visit counts
    if (pick < 0 && ply_in_game < cfg_.temperature_plies && total > 0) {
        std::uniform_int_distribution<uint32_t> dist(1, total);
        uint32_t roll = dist(rng);
        for (int i = 0; i < n.num; ++i) {
            if (roll <= visits[i]) { pick = i; break; }
            roll -= visits[i];
        }
        if (pick < 0) pick = n.num - 1;
    }

    // 3) greedy by visits
    if (pick < 0) {
        uint32_t bestv = 0;
        for (int i = 0; i < n.num; ++i)
            if (visits[i] > bestv) { bestv = visits[i]; pick = i; }
        if (pick < 0) pick = 0;   // all-zero fallback (shouldn't happen)
    }

    ch.move = es[pick].move;
    ch.chosen_edge = pick;
    ch.child_node = es[pick].child;
    return ch;
}

void Search::apply_root_move(uint16_t packed_move, int child_node) {
    Position np = make_move(hist_->back(), unpack_move(packed_move));
    hist_->push_back(std::move(np));

    // Reuse only fully-expanded subtrees: INFLIGHT children hold tasks that would
    // be orphaned by tasks_.clear(), wedging the new root in ST_INFLIGHT forever.
    bool reuse = false;
    if (child_node >= 0 && tree_.node(child_node).state == Tree::ST_EXPANDED) {
        root_ = child_node;
        reuse = true;
    }
    if (!reuse) {
        tree_.clear();
        root_ = tree_.new_node();
    }
    tasks_.clear();
    live_tasks_ = 0;
    // Fresh root noise at every move. Reused (already-expanded) roots never get
    // an evaluation task of their own, so without this they'd play the whole
    // move with un-noised priors.
    if (reuse) maybe_noise_root();
}

void Search::root_visit_distribution(std::vector<std::pair<int, int>>& out) const {
    out.clear();
    const Tree::Node& n = tree_.node(root_);
    const Tree::Edge* es = tree_.edges(root_);
    out.reserve(n.num);
    for (int i = 0; i < n.num; ++i)
        if (es[i].policy_idx >= 0 && es[i].visits > 0)
            out.emplace_back(es[i].policy_idx, static_cast<int>(es[i].visits));
}

}  // namespace chess
