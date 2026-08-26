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

    // self-play specific
    bool root_dirichlet = true;
    float dirichlet_eps = 0.25f;
    float dirichlet_alpha = 0.3f;
    int temperature_plies = 30;      // sample by visit counts before this ply, greedy after
    // Opening variety: for the first N plies sample moves from the
    // Dirichlet-noised PRIOR instead of visit counts. At self-play visit
    // budgets the visit distribution is mostly search noise; sampling the
    // prior diversifies games (anti draw-spiral) at zero eval cost.
    // 0 = off (match mode default). Must be <= temperature_plies to matter.
    int prior_plies = 0;
};

// A pending neural-network evaluation produced by the search; filled by the
// batching infrastructure, then handed back via complete_task().
struct EvalTask {
    EncodedPosition enc;
    std::vector<uint16_t> moves;        // packed legal moves at the leaf
    std::vector<float> priors;          // filled by evaluator: masked softmax, len == moves.size()
    float value = 0.0f;                 // scalarised WDL from leaf-stm perspective, filled by evaluator
    int32_t node = -1;                  // tree node awaiting this evaluation
    uint8_t stm = 0;                    // side to move at the leaf
    std::vector<std::pair<int32_t, int32_t>> path;   // descent path {node, edge} for backprop
};

class Tree {
public:
    static constexpr uint8_t ST_UNEXPANDED = 0;
    static constexpr uint8_t ST_INFLIGHT = 1;
    static constexpr uint8_t ST_EXPANDED = 2;
    static constexpr uint8_t ST_TERMINAL = 3;

    struct Edge {
        uint16_t move = 0;          // packed
        uint16_t prior = 0;         // 0..65535
        int16_t policy_idx = -1;    // action index into 4672
        uint32_t visits = 0;
        float value_sum = 0.0f;     // parent-perspective
        int32_t child = -1;         // materialized node index or -1
    };
    struct Node {
        int32_t first = -1;
        int32_t num = 0;
        uint8_t state = ST_UNEXPANDED;
        float term_value = 0.0f;
        int32_t task_slot = -1;
        uint8_t noised = 0;        // root Dirichlet noise already applied here
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

private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
};

// One search instance per active game position.
class Search {
public:
    Search(const MCTSConfig& cfg, std::vector<Position>&& game_history /*incl. current*/,
           std::mt19937_64* rng);

    void set_budget(int visits) { budget_ = visits; }
    int budget() const { return budget_; }
    int root_visits() const;
    float root_q() const;               // mean over root edges

    // One gathering round: up to max_leaves descents. New evaluations are stored in
    // tasks() and their slot indices reported.
    void gather_round(int max_leaves, std::vector<int>& out_task_slots);
    void complete_task(int slot);
    std::vector<EvalTask>& tasks() { return tasks_; }
    EvalTask& task(int slot) { return tasks_[slot]; }

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
    size_t pending_tasks() const { return live_tasks_; }

    // Apply the chosen move: push resulting position onto game history and rebase
    // the tree onto the chosen child subtree when available.
    void apply_root_move(uint16_t packed_move, int child_node);

private:
    struct PathEntry { int32_t node; int32_t edge; };

    static uint32_t node_visits(const Tree::Node& n, const Tree::Edge* es);
    float cpuct(float parent_visits) const;
    int select_child(int node, std::vector<PathEntry>& path);
    void backprop(const std::vector<PathEntry>& path, float value_from_leaf_stm);
    // Undo the virtual-loss subtraction of an aborted descent without counting it.
    void compensate_vl(const std::vector<PathEntry>& path);
    void descend_once(std::vector<int>& out_task_slots);
    void make_task(const Position& leaf, const std::vector<Position>& walk,
                   const std::vector<Move>& legal, int node_idx,
                   const std::vector<PathEntry>& path,
                   std::vector<int>& out_task_slots);
    void maybe_noise_root();

    MCTSConfig cfg_;
    std::unique_ptr<std::vector<Position>> hist_;
    Tree tree_;
    int root_ = -1;
    int budget_ = 800;
    std::mt19937_64* rng_ = nullptr;     // owned by caller (worker thread)
    std::vector<EvalTask> tasks_;
    size_t live_tasks_ = 0;
};

}  // namespace chess
