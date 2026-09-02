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

void Search::backprop(const std::vector<PathEntry>& path, float value_from_leaf_stm) {
    float contrib = -value_from_leaf_stm;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Tree::Edge& e = tree_.edges(it->node)[it->edge];
        e.visits += 1;
        e.value_sum += contrib + cfg_.virtual_loss;
        contrib = -contrib;
    }
}

EvalTask Search::make_task(const Position& leaf, const std::vector<Position>& walk,
                           const std::vector<Move>& legal, int node_idx,
                           const std::vector<PathEntry>& path) {
    EvalTask t;
    t.moves.reserve(legal.size());
    t.priors.assign(legal.size(), 0.0f);
    for (const Move& m : legal) t.moves.push_back(pack_move(m));
    t.stm = leaf.stm;
    t.root_stm = hist_->back().stm;
    t.path.reserve(path.size());
    for (const PathEntry& pe : path) t.path.emplace_back(pe.node, pe.edge);

    t.enc = encode_position(leaf, *hist_, walk);
    t.node = node_idx;
    tree_.node(node_idx).state = Tree::ST_INFLIGHT;
    return t;
}

void Search::compensate_vl(const std::vector<PathEntry>& path) {
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        Tree::Edge& e = tree_.edges(it->node)[it->edge];
        e.value_sum += cfg_.virtual_loss;
    }
}

void Search::descend_once(std::vector<EvalTask>& out_tasks) {
    thread_local std::vector<PathEntry> path;
    thread_local std::vector<Position> walk;
    path.clear();
    walk.clear();
    int node = root_;

    auto cur = [&]() -> const Position& { return walk.empty() ? hist_->back() : walk.back(); };

    for (;;) {
        Tree::Node& n = tree_.node(node);

        if (n.state == Tree::ST_TERMINAL) {
            backprop(path, n.term_value);
            return;
        }
        if (n.state == Tree::ST_INFLIGHT) {
            compensate_vl(path);
            return;
        }
        if (n.state == Tree::ST_UNEXPANDED) {
            const Position& leaf = cur();
            thread_local std::vector<Move> legal;
            legal.clear();
            gen_legal_moves(leaf, legal);
            if (legal.empty()) {
                bool check = attacks::square_attacked(
                    leaf, leaf.king_sq(static_cast<Color>(leaf.stm)),
                    static_cast<Color>(leaf.stm ^ 1), leaf.occ());
                n.state = Tree::ST_TERMINAL;
                n.term_value = check ? -1.0f : draw_value();
                backprop(path, n.term_value);
                return;
            }
            out_tasks.push_back(make_task(leaf, walk, legal, node, path));
            return;
        }

        int ei = select_child(node, path);
        Tree::Edge& e = tree_.edges(node)[ei];
        e.value_sum -= cfg_.virtual_loss;

        if (e.child < 0) e.child = tree_.new_node();

        Position next = make_move(cur(), unpack_move(e.move));

        if (cfg_.use_twofold_draw) {
            int seen = 0;
            for (const Position& p : *hist_)
                if (p.hash == next.hash) ++seen;
            for (const Position& p : walk)
                if (p.hash == next.hash) ++seen;
            if (seen >= 1) {   // was >= 2 (threefold); this is a search-internal twofold prune
                Tree::Node& cn = tree_.node(e.child);
                cn.state = Tree::ST_TERMINAL;
                cn.term_value = draw_value_for(next.stm);
                backprop(path, cn.term_value);
                return;
            }
        }
        if (next.halfmove >= 100) {
            Tree::Node& cn = tree_.node(e.child);
            cn.state = Tree::ST_TERMINAL;
            cn.term_value = draw_value_for(next.stm);
            backprop(path, cn.term_value);
            return;
        }

        walk.push_back(std::move(next));
        node = e.child;
    }
}

void Search::gather_round(int max_leaves, std::vector<EvalTask>& out_tasks) {
    for (int i = 0; i < max_leaves; ++i)
        descend_once(out_tasks);
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

void Search::complete_task(EvalTask&& t) {
    Tree::Node& n = tree_.node(t.node);
    if (n.state != Tree::ST_INFLIGHT) return;

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

    if (pick < 0 && ply_in_game < cfg_.temperature_plies && total > 0) {
        std::uniform_int_distribution<uint32_t> dist(1, total);
        uint32_t roll = dist(rng);
        for (int i = 0; i < n.num; ++i) {
            if (roll <= visits[i]) { pick = i; break; }
            roll -= visits[i];
        }
        if (pick < 0) pick = n.num - 1;
    }

    if (pick < 0) {
        uint32_t bestv = 0;
        for (int i = 0; i < n.num; ++i)
            if (visits[i] > bestv) { bestv = visits[i]; pick = i; }
        if (pick < 0) pick = 0;
    }

    ch.move = es[pick].move;
    ch.chosen_edge = pick;
    ch.child_node = es[pick].child;
    return ch;
}

void Search::apply_root_move(uint16_t packed_move, int child_node) {
    Position np = make_move(hist_->back(), unpack_move(packed_move));
    hist_->push_back(std::move(np));

    bool reuse = (child_node >= 0 && tree_.node(child_node).state == Tree::ST_EXPANDED);
    if (reuse) {
        tree_.keep_subtree(child_node);
        root_ = 0;   // keep_subtree always relabels the retained root to index 0
        maybe_noise_root();
    } else {
        tree_.clear();
        root_ = tree_.new_node();
    }
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

void Tree::keep_subtree(int new_root) {
    std::vector<Node> nn;
    std::vector<Edge> ne;
    nn.reserve(nodes_.size() / 4 + 64);
    ne.reserve(edges_.size() / 4 + 64);

    std::vector<int> old2new(nodes_.size(), -1);
    std::vector<int> stack;
    old2new[new_root] = 0;
    nn.push_back(nodes_[new_root]);
    stack.push_back(new_root);

    while (!stack.empty()) {
        int o = stack.back();
        stack.pop_back();
        int mapped = old2new[o];
        Node cp = nodes_[o];
        if (cp.num > 0) {
            const Edge* old_es = edges_.data() + cp.first;
            int32_t first_new = static_cast<int32_t>(ne.size());
            for (int i = 0; i < cp.num; ++i) {
                Edge e = old_es[i];
                if (e.child >= 0) {
                    if (old2new[e.child] < 0) {
                        old2new[e.child] = static_cast<int>(nn.size());
                        nn.push_back(nodes_[e.child]);
                        stack.push_back(e.child);
                    }
                    e.child = old2new[e.child];
                }
                ne.push_back(e);
            }
            cp.first = first_new;
        }
        nn[mapped] = cp;
    }
    nodes_.swap(nn);
    edges_.swap(ne);
}

}  // namespace chess