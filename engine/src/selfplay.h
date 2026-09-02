#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "mcts.h"
#include "nn_client.h"
#include <deque>

namespace chess {

struct SelfPlayConfig {
    int full_visits = 512;
    int fast_visits = 96;
    float fast_prob = 0.75f;
    int min_fresh_visits = 0;
    int leaves_per_round = 12;
    int batch_size = 256;
    int temperature_plies = 30;
    int prior_plies = 10;
    int max_plies = 450;
    float adjudicate_material_pawns = 4.0f;
    float contempt = 0.5f;
    float value_material_alpha = 0.f;
    bool resign_enabled = false;
    float resign_threshold = -0.90f;
    int resign_consecutive = 2;
    float resign_continue_frac = 0.05f;
    int resign_min_ply = 20;
    MCTSConfig mcts;
};

class EvalBatcher {
public:
    struct Item {
        int owner;
        Search* search;
        EvalTask task;
    };

    EvalBatcher(NNEvaluator& nn, int batch_size, float value_mat_alpha = 0.f,
                float contempt = 0.5f)
        : nn_(nn), batch_size_(batch_size), value_mat_alpha_(value_mat_alpha),
          contempt_(contempt) {}
    ~EvalBatcher() { stop(); }

    void set_mailboxes(int n);
    void start();
    void stop();
    void submit(int owner, Search* s, EvalTask&& t);
    void submit_many(int owner, Search* s, std::vector<EvalTask>& ts);

    size_t drain_wait(int owner, std::vector<Item>& out, int timeout_ms);
    uint64_t completions_seen() const { return completed_.load(std::memory_order_acquire); }

    std::atomic<uint64_t> evals_done{0};
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> routed{0};
    std::atomic<uint64_t> batches_run{0};
    std::atomic<int> last_batch_size{0};

private:
    void feeder_loop();

    NNEvaluator& nn_;
    int batch_size_;
    float value_mat_alpha_;
    float contempt_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> queue_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> completed_{0};

    struct Mailbox {
        std::mutex mu;
        std::condition_variable cv;
        std::vector<Item> q;
    };
    std::vector<std::unique_ptr<Mailbox>> mailboxes_;
};

struct Sample {
    EncodedPosition enc;
    std::vector<std::pair<int, int>> visit_dist;
    int material_diff = 0;
    uint8_t stm = 0;
};

struct FinishedGame {
    uint8_t result = 1;
    std::vector<Sample> samples;
};

class ShardWriter {
public:
    static constexpr int POLICY_SLOTS = 64;
    static constexpr size_t RECORD_BYTES =
        NUM_PLANES * 8 + SCALAR_COUNT * 4 + POLICY_SLOTS * 4 + 4;

    ShardWriter(std::string out_dir, std::string prefix, uint64_t model_hash,
                int games_per_shard, int start_id = 0)
        : out_dir_(std::move(out_dir)), prefix_(std::move(prefix)),
          model_hash_(model_hash), games_per_shard_(games_per_shard),
          next_shard_id_(start_id) {}

    void add_game(FinishedGame&& g);
    void flush_if_needed();
    void flush();

private:
    std::string out_dir_, prefix_;
    uint64_t model_hash_;
    int games_per_shard_;
    int buffered_games_ = 0;
    size_t buffered_games_positions_ = 0;
    size_t next_shard_id_ = 0;
    std::vector<char> buf_;

    static void put_u32(std::vector<char>& v, uint32_t x);
    static void put_u64(std::vector<char>& v, uint64_t x);
};

int run_selfplay(const std::string& model_path, const std::string& out_dir,
                 int total_games, int threads, SelfPlayConfig cfg,
                 uint64_t seed, int shard_games, bool prefer_gpu,
                 int games_per_worker = 4, int shard_start = 0);

std::string run_match(const std::string& model_a, const std::string& model_b,
                      int games, int visits, int batch_size, bool prefer_gpu,
                      int threads = 4, int games_per_worker = 3, const std::string& pgn_out = "");

int run_bench(const std::string& model_path, int seconds, int batch_size, bool prefer_gpu);

}  // namespace chess