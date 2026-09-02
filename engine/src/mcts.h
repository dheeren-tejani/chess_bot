#pragma once
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include "position.h"
#include "encoder.h"
#include "policy_map.h"

namespace chess {

struct MCTSConfig {
    float cpuct_init = 1.25f;
    float cpuct_base = 19652.0f;
    float fpu_reduction = 0.2f;
    float virtual_loss = 1.0f;
    bool use_twofold_draw = true;
    float contempt = 0.5f;

    bool root_dirichlet = true;
    float dirichlet_eps = 0.25f;
    float dirichlet_alpha = 0.3f;
    int temperature_plies = 30;
    int prior_plies = 0;
};

struct EvalTask {
    EncodedPosition enc;
    std::vector<uint16_t> moves;
    std::vector<float> priors;
    float value = 0.0f;
    int32_t node = -1;
    uint8_t stm = 0;
    uint8_t root_stm = 0;
    std::vector<std::pair<int32_t, int32_t>> path;
};

struct PathEntry {
    int32_t node;
    int32_t edge;
};

class Tree {
public:
    static constexpr uint8_t ST_UNEXPANDED = 0;
    static constexpr uint8_t ST_INFLIGHT = 1;
    static constexpr uint8_t ST_EXPANDED = 2;
    static constexpr uint8_t ST_TERMINAL = 3;

    struct Edge {
        uint16_t move = 0;
        uint16_t prior = 0;
        int16_t policy_idx = -1;
        uint32_t visits = 0;
        float value_sum = 0.0f;
        int32_t child = -1;
    };
    struct Node {
        int32_t first = -1;
        int32_t num = 0;
        uint8_t state = ST_UNEXPANDED;
        float term_value = 0.0f;
        int32_t task_slot = -1;
        uint8_t noised = 0;
    };

    int new_node() {
        nodes_.push_back(Node{});
        return static_cast<int>(nodes_.size() - 1);
    }
    Node& node(int i) { return nodes_[i]; }
    const Node& node(int i) const { return nodes_[i]; }
    Edge* edges(int node) { return edges_.data() + nodes_[node].first; }
    const Edge* edges(int node) const { return edges_.data() + nodes_[node].first; }

    template <typename Vec>
    void attach_edges(int node, Vec&& es) {
        nodes_[node].first = static_cast<int32_t>(edges_.size());
        nodes_[node].num = static_cast<int32_t>(es.size());
        for (const auto& e : es) edges_.push_back(e);
    }
    void clear() { nodes_.clear(); edges_.clear(); }
    size_t node_count() const { return nodes_.size(); }
    size_t edge_count() const { return edges_.size(); }
    void keep_subtree(int new_root);

private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

class Search {
public:
    Search(const MCTSConfig& cfg, std::vector<Position>&& game_history,
           std::mt19937_64* rng);

    void set_budget(int visits) { budget_ = visits; }
    int budget() const { return budget_; }
    int root_visits() const;
    float root_q() const;
    float draw_value() const { return 2.0f * cfg_.contempt - 1.0f; }  // draw value for the ROOT player
    // Draw score from the perspective of `node_stm` (parity-corrected contempt):
    float draw_value_for(uint8_t node_stm) const {
        return (node_stm == hist_->back().stm) ? draw_value() : -draw_value();
    }

    void gather_round(int max_leaves, std::vector<EvalTask>& out_tasks);
    void complete_task(EvalTask&& t);

    struct Choice {
        uint16_t move = 0;
        int child_node = -1;
        int chosen_edge = -1;
    };
    Choice pick_move(std::mt19937_64& rng, int ply_in_game);

    void root_visit_distribution(std::vector<std::pair<int, int>>& out) const;
    const Position& root_position() const { return hist_->back(); }
    std::vector<Position>& history() { return *hist_; }
    Tree& tree() { return tree_; }
    const Tree& tree() const { return tree_; }
    int root() const { return root_; }

    void apply_root_move(uint16_t packed_move, int child_node);

private:
    static uint32_t node_visits(const Tree::Node& n, const Tree::Edge* es);
    float cpuct(float parent_visits) const;
    int select_child(int node, std::vector<PathEntry>& path);
    void backprop(const std::vector<PathEntry>& path, float value_from_leaf_stm);
    void compensate_vl(const std::vector<PathEntry>& path);
    void descend_once(std::vector<EvalTask>& out_tasks);
    EvalTask make_task(const Position& leaf, const std::vector<Position>& walk,
                       const std::vector<Move>& legal, int node_idx,
                       const std::vector<PathEntry>& path);
    void maybe_noise_root();

    MCTSConfig cfg_;
    std::unique_ptr<std::vector<Position>> hist_;
    Tree tree_;
    int root_ = -1;
    int budget_ = 800;
    std::mt19937_64* rng_ = nullptr;
};

}  // namespace chess