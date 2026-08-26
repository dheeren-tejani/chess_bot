#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "mcts.h"
#include "nn_client.h"

namespace chess {

struct SelfPlayConfig {
    int full_visits = 512;
    int fast_visits = 96;
    float fast_prob = 0.75f;
    int leaves_per_round = 12;
    int batch_size = 256;
    int temperature_plies = 30;
    // Sample the first N plies from the (noised) prior instead of visit
    // counts - opening variety without extra eval cost. 0 disables.
    int prior_plies = 10;
    int max_plies = 450;
    // Anti draw-death: a game that reaches the ply cap is NOT scored as a draw
    // by default - if |material diff| >= adjudicate_material_pawns the leader
    // is scored as the winner (value net gets decisive labels). Values <= 0
    // restore pure-draw adjudication.
    float adjudicate_material_pawns = 4.0f;
    // Contempt: value of a draw inside the SEARCH only (training labels stay
    // honest 0.5). 0.5 = neutral. Lower (e.g. 0.45) makes both players avoid
    // drifting into repetition/sterile draws - the classical anti-draw-death
    // tool. Pairs with use_twofold_draw=true so repetitions are both VISIBLE
    // (terminal 0.0) and unattractive (< 0.45 uncertainty).
    float contempt = 0.5f;
    // Auxiliary material signal for the search value (cold-start aid).
    // value = wdl_expected + alpha * (material_pred / 8). The WDL head learns
    // nothing from 90%+ draw data (it correctly concludes "nothing matters"),
    // which freezes the search at accident-decided games. The material head
    // HAS dense supervision, so blending a little of it into the search value
    // gives self-play a gradient toward converting material until the WDL
    // head discriminates on its own (probe: ev(queen-up) >> ev(start)).
    // 0 = off. Matches/gate always use 0.
    float value_material_alpha = 0.f;
    bool resign_enabled = false;
    float resign_threshold = -0.90f;
    int resign_consecutive = 2;
    float resign_continue_frac = 0.05f;
    int resign_min_ply = 20;
    MCTSConfig mcts;
};

// Shared evaluation pipeline: workers submit (owner, search, slot) items; a feeder
// thread batches them through the NNEvaluator, fills priors/value into the tasks,
// and routes them to the owning worker's mailbox. Only the OWNING worker applies
// completions to its trees (single-writer-per-tree => no locking in MCTS).
class EvalBatcher {
public:
    struct Item { int owner; Search* search; int slot; };

    EvalBatcher(NNEvaluator& nn, int batch_size, float value_mat_alpha = 0.f,
                float contempt = 0.5f)
        : nn_(nn), batch_size_(batch_size), value_mat_alpha_(value_mat_alpha),
          contempt_(contempt) {}
    ~EvalBatcher() { stop(); }

    void set_mailboxes(int n);
    void start();
    void stop();
    void submit(int owner, Search* s, int slot);

    // Drain routed completions for this worker; if empty, block up to timeout_ms.
    size_t drain_wait(int owner, std::vector<std::pair<Search*, int>>& out,
                      int timeout_ms);
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
    std::vector<Item> queue_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> completed_{0};

    struct Mailbox {
        std::mutex mu;
        std::condition_variable cv;
        std::vector<std::pair<Search*, int>> q;
    };
    std::vector<std::unique_ptr<Mailbox>> mailboxes_;
};

// One finished training sample.
struct Sample {
    EncodedPosition enc;
    std::vector<std::pair<int, int>> visit_dist;   // policy_idx -> visits
    int material_diff = 0;                          // stm perspective
    uint8_t stm = 0;                                // 0 white, 1 black
};

struct FinishedGame {
    uint8_t result = 1;              // 0 white wins, 1 draw, 2 black wins
    std::vector<Sample> samples;
};

// Binary shard writer (gzip). Record layout (1200 bytes, little-endian):
//   planes  : uint64[113]           (904 B; 112 age-block planes + 1 en-passant)
//   scalars : float[9]              ( 36 B)
//   policy  : {u16 idx, u16 visits}[64], idx 0xFFFF = unused   (256 B)
//   meta    : u8 result, u16 material_f16_bits, u8 stm      (  4 B)
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
    void flush();                    // writes pending games as one gz shard

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

// Run self-play generation. Returns process exit code.
// `shard_start` continues shard numbering from an interrupted previous launch
// writing into the same directory (engine numbering restarts at 0 by default,
// which would overwrite existing shards).
int run_selfplay(const std::string& model_path, const std::string& out_dir,
                 int total_games, int threads, SelfPlayConfig cfg,
                 uint64_t seed, int shard_games, bool prefer_gpu,
                 int games_per_worker = 4, int shard_start = 0);

// Parallel engine-vs-engine match. Returns JSON summary string.
std::string run_match(const std::string& model_a, const std::string& model_b,
                      int games, int visits, int batch_size, bool prefer_gpu,
                      int threads = 4, int games_per_worker = 3);

// Quick evaluator throughput benchmark.
int run_bench(const std::string& model_path, int seconds, int batch_size, bool prefer_gpu);

}  // namespace chess
