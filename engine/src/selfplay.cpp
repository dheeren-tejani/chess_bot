#include "selfplay.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <random>
#include <fstream>
#include <zlib.h>
#include <map>

namespace chess {

void EvalBatcher::set_mailboxes(int n) {
    mailboxes_.clear();
    for (int i = 0; i < n; ++i) mailboxes_.push_back(std::make_unique<Mailbox>());
}

void EvalBatcher::start() {
    running_.store(true);
    thread_ = std::thread([this] { feeder_loop(); });
}

void EvalBatcher::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void EvalBatcher::submit(int owner, Search* s, EvalTask&& t) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        queue_.push_back({owner, s, std::move(t)});
    }
    submitted.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_one();
}

void EvalBatcher::submit_many(int owner, Search* s, std::vector<EvalTask>& ts) {
    if (ts.empty()) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& t : ts) queue_.push_back({owner, s, std::move(t)});
    }
    submitted.fetch_add(ts.size(), std::memory_order_relaxed);
    cv_.notify_one();
}

size_t EvalBatcher::drain_wait(int owner, std::vector<Item>& out, int timeout_ms) {
    Mailbox& mb = *mailboxes_[owner];
    std::unique_lock<std::mutex> lk(mb.mu);
    if (mb.q.empty() && timeout_ms > 0)
        mb.cv.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                       [&] { return !mb.q.empty(); });
    out.insert(out.end(), std::make_move_iterator(mb.q.begin()), std::make_move_iterator(mb.q.end()));
    size_t k = mb.q.size();
    mb.q.clear();
    return k;
}

void EvalBatcher::feeder_loop() {
    std::vector<Item> local;
    std::vector<uint64_t> planes(static_cast<size_t>(batch_size_) * NUM_PLANES);
    std::vector<float> scalars(static_cast<size_t>(batch_size_) * SCALAR_COUNT, 0.f);
    std::vector<float> policy, wdl, mat;
    std::vector<int> idx;

    auto fill_task = [&](EvalTask& t, int row) {
        const float* prow = policy.data() + static_cast<size_t>(row) * POLICY_ACTIONS;
        const size_t K = t.moves.size();
        float maxl = -1e30f;
        idx.resize(K);
        for (size_t k = 0; k < K; ++k) {
            Move m = unpack_move(t.moves[k]);
            idx[k] = move_policy_index(m, static_cast<Color>(t.stm));
            maxl = std::max(maxl, prow[idx[k]]);
        }
        double sum = 0;
        for (size_t k = 0; k < K; ++k) {
            t.priors[k] = std::exp(prow[idx[k]] - maxl);
            sum += t.priors[k];
        }
        for (size_t k = 0; k < K; ++k) t.priors[k] /= static_cast<float>(sum);

        const float* wrow = wdl.data() + static_cast<size_t>(row) * 3;
        float wmax = std::max({wrow[0], wrow[1], wrow[2]});
        double e0 = std::exp(wrow[0] - wmax), e1 = std::exp(wrow[1] - wmax),
               e2 = std::exp(wrow[2] - wmax);
        double es = e0 + e1 + e2;
        
        double c_eff = (t.stm == t.root_stm) ? static_cast<double>(contempt_)
                                              : (1.0 - static_cast<double>(contempt_));
        double base = (e0 + c_eff * e1) / es;
        base = 2.0 * base - 1.0;

        if (value_mat_alpha_ != 0.f) {
            // train.py trains the material head on target = material/8, so `m` is
            // already in that scale — don't divide again.
            double m = static_cast<double>(mat[static_cast<size_t>(row)]);
            base += value_mat_alpha_ * m;
        }
        t.value = static_cast<float>(std::max(-1.0, std::min(1.0, base)));
    };

    for (;;) {
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::milliseconds(2),
                         [&] { return !queue_.empty() || !running_.load(); });
            if (queue_.empty()) {
                if (!running_.load()) return;
                continue;
            }
            if (queue_.size() < static_cast<size_t>(batch_size_)) {
                auto deadline = std::chrono::steady_clock::now() +
                                std::chrono::milliseconds(4);
                cv_.wait_until(lk, deadline, [&] {
                    return queue_.size() >= static_cast<size_t>(batch_size_);
                });
            }
            size_t take = std::min(queue_.size(), static_cast<size_t>(batch_size_));
            local.assign(std::make_move_iterator(queue_.begin()),
                         std::make_move_iterator(queue_.begin() + take));
            queue_.erase(queue_.begin(), queue_.begin() + take);
        }

        const size_t n = local.size();
        for (size_t b = 0; b < n; ++b) {
            const EvalTask& t = local[b].task;
            std::copy(t.enc.planes.begin(), t.enc.planes.end(),
                      planes.begin() + b * NUM_PLANES);
            std::copy(t.enc.scalars.begin(), t.enc.scalars.end(),
                      scalars.begin() + b * SCALAR_COUNT);
        }

        nn_.evaluate(planes.data(), scalars.data(), static_cast<int>(n), policy, wdl, mat);

        for (size_t i = 0; i < n; ++i) {
            fill_task(local[i].task, static_cast<int>(i));
        }

        for (size_t i = 0; i < n; ++i) {
            int owner = local[i].owner;
            Mailbox& mb = *mailboxes_[owner];
            {
                std::lock_guard<std::mutex> lk(mb.mu);
                mb.q.push_back(std::move(local[i]));
            }
            mailboxes_[owner]->cv.notify_one();
        }

        routed.fetch_add(n, std::memory_order_relaxed);
        batches_run.fetch_add(1, std::memory_order_relaxed);
        last_batch_size.store(static_cast<int>(n), std::memory_order_relaxed);
        evals_done.fetch_add(n, std::memory_order_relaxed);
        completed_.fetch_add(n, std::memory_order_release);
        cv_.notify_all();

        local.clear();
    }
}

static void put_u16(std::vector<char>& v, uint16_t x) {
    v.push_back(char(x & 0xFF));
    v.push_back(char((x >> 8) & 0xFF));
}
void ShardWriter::put_u32(std::vector<char>& v, uint32_t x) {
    for (int i = 0; i < 4; ++i) v.push_back(char((x >> (8 * i)) & 0xFF));
}
void ShardWriter::put_u64(std::vector<char>& v, uint64_t x) {
    for (int i = 0; i < 8; ++i) v.push_back(char((x >> (8 * i)) & 0xFF));
}

void ShardWriter::add_game(FinishedGame&& g) {
    for (const Sample& s : g.samples) {
        for (uint64_t w : s.enc.planes) put_u64(buf_, w);
        for (float f : s.enc.scalars) {
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            put_u32(buf_, bits);
        }
        std::vector<std::pair<int, int>> dist = s.visit_dist;
        std::sort(dist.begin(), dist.end(),
                  [](auto& a, auto& b) { return a.second > b.second; });
        int filled = 0;
        for (auto& [idx, visits] : dist) {
            if (filled >= POLICY_SLOTS) break;
            put_u16(buf_, static_cast<uint16_t>(idx));
            put_u16(buf_, static_cast<uint16_t>(std::min(65535, visits)));
            ++filled;
        }
        for (; filled < POLICY_SLOTS; ++filled) { put_u16(buf_, 0xFFFF); put_u16(buf_, 0); }

        buf_.push_back(char(g.result));
        auto to_f16 = [](float f) -> uint16_t {
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
            uint32_t man = bits & 0x7FFFFF;
            if (exp <= 0) return static_cast<uint16_t>(sign);
            if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
            return static_cast<uint16_t>(sign | (exp << 10) | (man >> 13));
        };
        float mf = static_cast<float>(s.material_diff);
        uint16_t m16 = to_f16(mf);
        put_u16(buf_, m16);
        buf_.push_back(char(s.stm));

        ++buffered_games_positions_;
    }
    ++buffered_games_;
    flush_if_needed();
}

void ShardWriter::flush_if_needed() {
    if (buffered_games_ >= games_per_shard_) flush();
}

void ShardWriter::flush() {
    if (buffered_games_ == 0) return;

    std::filesystem::create_directories(out_dir_);
    char name[256];
    snprintf(name, sizeof(name), "%s/%s_%06lu.gz", out_dir_.c_str(), prefix_.c_str(),
             static_cast<unsigned long>(next_shard_id_++));

    std::string tmp_name = std::string(name) + ".tmp";
    gzFile f = gzopen(tmp_name.c_str(), "wb1");
    if (!f) {
        fprintf(stderr, "FATAL cannot open shard %s\n", tmp_name.c_str());
        exit(1);
    }
    std::vector<char> header;
    header.push_back('C'); header.push_back('B'); header.push_back('S'); header.push_back('P');
    put_u32(header, 1);
    put_u32(header, static_cast<uint32_t>(buffered_games_));
    put_u64(header, model_hash_);
    gzwrite(f, header.data(), static_cast<unsigned>(header.size()));
    gzwrite(f, buf_.data(), static_cast<unsigned>(buf_.size()));
    gzclose(f);
    std::filesystem::rename(tmp_name, name);

    fprintf(stdout,
            "{\"type\":\"shard\",\"path\":\"%s\",\"games\":%d,\"positions\":%zu}\n",
            name, buffered_games_, buffered_games_positions_);
    fflush(stdout);

    buf_.clear();
    buffered_games_ = 0;
    buffered_games_positions_ = 0;
}

static uint64_t fnv1a_file(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) return 0;
    uint64_t h = 0xCBF29CE484222325ULL;
    std::vector<char> tmp(1 << 20);
    size_t n;
    while ((n = fread(tmp.data(), 1, tmp.size(), fp)) > 0)
        for (size_t i = 0; i < n; ++i) {
            h ^= static_cast<unsigned char>(tmp[i]);
            h *= 0x100000001B3ULL;
        }
    fclose(fp);
    return h;
}

struct SharedStats {
    std::atomic<uint64_t> games_done{0};
    std::atomic<uint64_t> positions_written{0};
    std::atomic<int64_t> total_plies{0};
};

struct GameSlot {
    std::unique_ptr<Search> search;
    FinishedGame game;
    int ply = 0;
    int budget = 0;
    int visits_base = 0;
    size_t inflight = 0;
    bool move_ready = false;
    bool finished = false;
    bool exhausted = false;
    bool resigned = false;
    int resign_streak = 0;
    int resign_side = -1;
};

static bool conclude_move(SelfPlayConfig& cfg, GameSlot& g, std::mt19937_64& rng,
                          const char** end_reason) {
    Search& s = *g.search;
    const Position& cur = s.root_position();

    // 1) Mate/stalemate check first
    const Tree::Node& rn = s.tree().node(s.root());
    if (rn.state == Tree::ST_TERMINAL || rn.num == 0) {
        std::vector<Move> legal;
        gen_legal_moves(cur, legal);
        if (legal.empty()) {
            bool check = attacks::square_attacked(
                cur, cur.king_sq(static_cast<Color>(cur.stm)),
                static_cast<Color>(cur.stm ^ 1), cur.occ());
            g.game.result = check ? ((cur.stm == WHITE) ? 2 : 0) : 1;
            *end_reason = check ? "mate" : "stalemate";
        } else {
            g.game.result = 1;
            *end_reason = rn.state == Tree::ST_TERMINAL ? "root-terminal-nonempty"
                                                        : "root-empty-expanded";
        }
        return true;
    }

    // 2) Rule-based draws / adjudications
    if (cur.halfmove >= 100) {
        g.game.result = 1;
        *end_reason = "50move";
        return true;
    }
    if (insufficient_material(cur)) {
        g.game.result = 1;
        *end_reason = "material";
        return true;
    }
    if (g.ply >= cfg.max_plies) {
        if (cfg.adjudicate_material_pawns > 0.f) {
            int d = material_diff_stm(cur);
            int white_diff = (cur.stm == WHITE) ? d : -d;
            if (std::abs(white_diff) >= static_cast<int>(cfg.adjudicate_material_pawns)) {
                g.game.result = (white_diff > 0) ? 0 : 2;
                *end_reason = "cap-material";
            } else {
                g.game.result = 1;
                *end_reason = "maxplies";
            }
        } else {
            g.game.result = 1;
            *end_reason = "maxplies";
        }
        return true;
    }

    // 3) Resignation check (per-side streak tracking)
    if (cfg.resign_enabled && g.ply >= cfg.resign_min_ply) {
        float q = s.root_q(); // from current stm's perspective
        if (q < cfg.resign_threshold) {
            if (g.resign_side == static_cast<int>(cur.stm)) {
                ++g.resign_streak;
            } else {
                g.resign_side = static_cast<int>(cur.stm);
                g.resign_streak = 1;
            }

            if (g.resign_streak >= cfg.resign_consecutive &&
                std::uniform_real_distribution<float>(0.f, 1.f)(rng) >=
                    cfg.resign_continue_frac) {
                // If White (0) resigns, Black (2) wins; if Black (1) resigns, White (0) wins
                g.game.result = (cur.stm == WHITE) ? 2 : 0;
                g.resigned = true;
                *end_reason = "resign";
                return true;
            }
        } else if (g.resign_side == static_cast<int>(cur.stm)) {
            // Only the streak's owner can reset their own streak
            g.resign_streak = 0;
            g.resign_side = -1;
        }
    }

    // 4) Record training sample
    Sample smp;
    s.root_visit_distribution(smp.visit_dist);
    smp.enc = encode_position(cur, s.history());
    smp.material_diff = material_diff_stm(cur);
    smp.stm = cur.stm;
    g.game.samples.push_back(std::move(smp));

    // 5) Pick move & apply
    Search::Choice ch = s.pick_move(rng, g.ply);
    s.apply_root_move(ch.move, ch.child_node);
    g.visits_base = s.root_visits();
    ++g.ply;

    const std::vector<Position>& h = s.history();
    uint64_t hh = h.back().hash;
    int seen = 0;
    for (const Position& p : h)
        if (p.hash == hh) ++seen;
    if (seen >= 3) {
        g.game.result = 1;
        *end_reason = "threefold";
        return true;
    }
    *end_reason = "ongoing";
    return false;
}

int run_selfplay(const std::string& model_path, const std::string& out_dir,
                 int total_games, int threads, SelfPlayConfig cfg,
                 uint64_t seed, int shard_games, bool prefer_gpu,
                 int games_per_worker, int shard_start) {
    attacks::init();

    NNEvaluator nn;
    std::string err;
    if (!nn.load(model_path, cfg.batch_size, prefer_gpu, &err)) {
        fprintf(stderr, "FATAL model load failed: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "{\"type\":\"info\",\"gpu\":%s,\"batch\":%d}\n",
            nn.is_gpu() ? "true" : "false", cfg.batch_size);

    EvalBatcher batcher(nn, cfg.batch_size, cfg.value_material_alpha, cfg.contempt);
    batcher.set_mailboxes(threads);
    ShardWriter writer(out_dir, "shard", fnv1a_file(model_path), shard_games, shard_start);
    SharedStats stats;
    std::atomic<int> next_game{0};
    std::mutex writer_mu;

    std::mutex end_mu;
    std::map<std::string, uint64_t> end_reasons;
    std::atomic<uint64_t> res_white{0}, res_draw{0}, res_black{0}, res_resigned{0};

    auto record_end = [&](const char* why, uint8_t result, bool resigned) {
        {
            std::lock_guard<std::mutex> lk(end_mu);
            ++end_reasons[why];
        }
        if (result == 0) res_white.fetch_add(1);
        else if (result == 1) res_draw.fetch_add(1);
        else res_black.fetch_add(1);
        if (resigned) res_resigned.fetch_add(1);
    };

    auto t_start = std::chrono::steady_clock::now();
    std::atomic<bool> stop{false};

    batcher.start();

    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ULL + t + 1);
            std::vector<GameSlot> slots(games_per_worker);
            std::vector<EvalBatcher::Item> ready;
            std::vector<EvalTask> ts;                 // NEW — hoisted out of the slot loop
            ts.reserve(static_cast<size_t>(cfg.leaves_per_round));

            auto start_new = [&](GameSlot& g) {
                std::vector<Position> hist(1);
                hist[0].set_from_fen(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                g.search = std::make_unique<Search>(cfg.mcts, std::move(hist), &rng);
                g.game = FinishedGame{};
                g.ply = 0;
                g.visits_base = 0;
                g.inflight = 0;
                g.move_ready = false;
                g.finished = false;
                g.exhausted = false;
                g.resigned = false;
                g.resign_streak = 0;
                g.resign_side = -1;
                float roll = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
                g.budget = (roll < cfg.fast_prob) ? cfg.fast_visits : cfg.full_visits;
                g.search->set_budget(g.budget);
            };
            auto try_claim = [&](GameSlot& g) {
                int gid = next_game.fetch_add(1);
                if (gid < total_games && !stop.load()) {
                    start_new(g);
                    return true;
                }
                g.finished = true;
                g.exhausted = true;
                return false;
            };

            for (auto& g : slots) try_claim(g);

            int wait_ms = 0;
            for (;;) {
                ready.clear();
                size_t got = batcher.drain_wait(t, ready, wait_ms);
                for (auto& item : ready) {
                    for (auto& g : slots) {
                        if (!g.search || g.exhausted) continue;
                        if (g.search.get() == item.search) { g.inflight -= 1; break; }
                    }
                    item.search->complete_task(std::move(item.task));
                }

                bool any_progress = got > 0;
                bool all_exhausted = true;

                for (auto& g : slots) {
                    if (g.exhausted) continue;
                    all_exhausted = false;

                    if (g.move_ready) {
                        const char* why = "?";
                        bool ended = conclude_move(cfg, g, rng, &why);
                        g.move_ready = false;
                        any_progress = true;
                        if (ended) {
                            record_end(why, g.game.result, g.resigned);
                            g.finished = true;
                        } else {
                            float roll =
                                std::uniform_real_distribution<float>(0.f, 1.f)(rng);
                            g.budget =
                                (roll < cfg.fast_prob) ? cfg.fast_visits : cfg.full_visits;
                            g.search->set_budget(g.budget);
                        }
                        continue;
                    }

                    // Budget + minimum-fresh-visits gate
                    const Tree::Node& rn = g.search->tree().node(g.search->root());
                    int rv = static_cast<int>(g.search->root_visits());
                    int fresh = rv - g.visits_base;
                    int need_fresh = std::min(cfg.min_fresh_visits, g.budget);
                    bool budget_met = (rv >= g.budget) && (fresh >= need_fresh);

                    if (rn.state == Tree::ST_TERMINAL || (g.inflight == 0 && budget_met)) {
                        g.move_ready = true;
                        continue;
                    }

                    // Gather while either the total budget OR the fresh-visits
                    // requirement is short. Without this second clause, a reused
                    // subtree that already exceeds budget+leaves_per_round in
                    // total visits stops gathering forever while fresh sits at 0
                    // (livelock: move_ready never becomes true).
                    bool need_more_total = (rv + static_cast<int>(g.inflight)) < (g.budget + cfg.leaves_per_round);
                    bool need_more_fresh = (fresh + static_cast<int>(g.inflight)) < need_fresh;
                    if (need_more_total || need_more_fresh) {
                        ts.clear();
                        g.search->gather_round(cfg.leaves_per_round, ts);
                        if (!ts.empty()) {
                            g.inflight += ts.size();
                            batcher.submit_many(t, g.search.get(), ts);
                            any_progress = true;
                        }
                    }
                }

                for (auto& g : slots) {
                    if (g.finished && !g.exhausted) {
                        size_t nsamples = g.game.samples.size();
                        {
                            std::lock_guard<std::mutex> lk(writer_mu);
                            writer.add_game(std::move(g.game));
                        }
                        stats.games_done.fetch_add(1);
                        stats.total_plies.fetch_add(nsamples);
                        try_claim(g);
                        any_progress = true;
                    }
                }

                if (all_exhausted) break;

                // Shutdown drain: after a global stop, exit only once every slot's
                // in-flight evaluations have come home (otherwise the feeder would
                // touch freed Search objects on its final batches).
                if (stop.load()) {
                    bool any_inflight = false;
                    for (auto& g : slots)
                        if (!g.exhausted && g.inflight > 0) { any_inflight = true; break; }
                    if (!any_inflight) break;
                }

                wait_ms = any_progress ? 0 : 5;
            }
        });
    }

    uint64_t last_evals = 0;
    auto last_t = t_start;
    while (!stop.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(3));
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last_t).count();
        uint64_t ev = batcher.evals_done.load();
        double eps = dt > 0 ? double(ev - last_evals) / dt : 0.0;
        last_evals = ev;
        last_t = now;
        double el = std::chrono::duration<double>(now - t_start).count();
        uint64_t gd = stats.games_done.load();
        double gph = el > 0 ? gd * 3600.0 / el : 0.0;
        fprintf(stderr,
                "{\"type\":\"stats\",\"elapsed_s\":%.0f,\"games\":%lu,\"total\":%d,"
                "\"evals_total\":%lu,\"evals_per_s\":%.0f,\"games_per_h\":%.1f,"
                "\"avg_len\":%.1f,\"submitted\":%lu,\"routed\":%lu,\"batches\":%lu,\"last_bsz\":%d}\n",
                el, (unsigned long)gd, total_games, (unsigned long)ev, eps, gph,
                gd ? double(stats.total_plies.load()) / double(gd) : 0.0,
                (unsigned long)batcher.submitted.load(),
                (unsigned long)batcher.routed.load(),
                (unsigned long)batcher.batches_run.load(),
                batcher.last_batch_size.load());
        if (gd >= static_cast<uint64_t>(total_games)) stop.store(true);
    }

    for (auto& th : workers) th.join();
    batcher.stop();
    {
        std::lock_guard<std::mutex> lk(writer_mu);
        writer.flush();
    }

    {
        uint64_t gd = stats.games_done.load();
        uint64_t plies = stats.total_plies.load();
        std::string reasons;
        {
            std::lock_guard<std::mutex> lk(end_mu);
            for (const auto& [k, v] : end_reasons) {
                if (!reasons.empty()) reasons += ",";
                reasons += "\"" + k + "\":" + std::to_string(v);
            }
        }
        fprintf(stdout,
                "{\"type\":\"selfplay_summary\",\"games\":%lu,\"positions\":%lu,"
                "\"avg_plies\":%.1f,\"white_wins\":%lu,\"draws\":%lu,\"black_wins\":%lu,"
                "\"resigned\":%lu,\"end_reasons\":{%s}}\n",
                (unsigned long)gd, (unsigned long)plies,
                gd ? double(plies) / double(gd) : 0.0,
                (unsigned long)res_white.load(), (unsigned long)res_draw.load(),
                (unsigned long)res_black.load(), (unsigned long)res_resigned.load(),
                reasons.c_str());
        fflush(stdout);
    }

    return 0;
}

std::string run_match(const std::string& model_a, const std::string& model_b,
                      int games, int visits, int batch_size, bool prefer_gpu,
                      int threads, int games_per_worker, const std::string& pgn_out) {
    attacks::init();
    MCTSConfig mc;
    mc.root_dirichlet = false;
    mc.temperature_plies = 2;

    struct MatchGame {
        std::unique_ptr<Search> search;
        bool a_is_white = true;
        int ply = 0;
        int result = -1;
        int game_id = 0;
        std::vector<std::string> moves;
        size_t inflight = 0;
        bool move_ready = false;
        bool finished = false;
        bool exhausted = false;
    };

    NNEvaluator nn_a, nn_b;
    std::string err;
    if (!nn_a.load(model_a, batch_size, prefer_gpu, &err))
        return std::string("{\"error\":\"A: ") + err + "\"}";
    if (!nn_b.load(model_b, batch_size, prefer_gpu, &err))
        return std::string("{\"error\":\"B: ") + err + "\"}";

    EvalBatcher batch_a(nn_a, batch_size);
    EvalBatcher batch_b(nn_b, batch_size);
    batch_a.set_mailboxes(threads);
    batch_b.set_mailboxes(threads);
    std::atomic<int> next_game{0};
    std::atomic<int> finished_games{0};
    std::atomic<int> aw{0}, ad{0}, al{0};

    std::mutex pgn_mu;
    if (!pgn_out.empty()) {
        std::ofstream init_file(pgn_out, std::ios::trunc);
    }

    batch_a.start();
    batch_b.start();

    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            std::mt19937_64 rng(0x6D7A4B3ULL * (t + 1));
            std::vector<MatchGame> slots(games_per_worker);
            std::vector<EvalBatcher::Item> ready;

            auto start_new = [&](MatchGame& g, int gid) {
                std::vector<Position> hist(1);
                hist[0].set_from_fen(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                g.search = std::make_unique<Search>(mc, std::move(hist), &rng);
                g.search->set_budget(visits);
                g.a_is_white = (gid % 2 == 0);
                g.ply = 0;
                g.result = -1;
                g.game_id = gid;
                g.moves.clear();
                g.inflight = 0;
                g.move_ready = false;
                g.finished = false;
                g.exhausted = false;
            };
            auto try_claim = [&](MatchGame& g) {
                int gid = next_game.fetch_add(1);
                if (gid < games) { start_new(g, gid); return true; }
                g.finished = true;
                g.exhausted = true;
                return false;
            };
            for (auto& g : slots) try_claim(g);

            auto conclude = [&](MatchGame& g) -> bool {
                Search& s = *g.search;
                const Position& cur = s.root_position();
                
                const Tree::Node& rn = s.tree().node(s.root());
                std::vector<Move> legal;
                gen_legal_moves(cur, legal);
                if (rn.state == Tree::ST_TERMINAL || rn.num == 0 || legal.empty()) {
                    g.result = legal.empty()
                        ? (attacks::square_attacked(
                               cur, cur.king_sq(static_cast<Color>(cur.stm)),
                               static_cast<Color>(cur.stm ^ 1), cur.occ())
                               ? ((cur.stm == WHITE) ? 2 : 0)
                               : 1)
                        : 1;
                    return true;
                }

                if (cur.halfmove >= 100 || insufficient_material(cur) || g.ply >= 300) {
                    g.result = 1;
                    return true;
                }

                uint64_t hh = s.history().back().hash;
                int seen = 0;
                for (const Position& p : s.history())
                    if (p.hash == hh) ++seen;
                if (seen >= 3) { g.result = 1; return true; }

                Search::Choice ch = s.pick_move(rng, g.ply);

                Move m = unpack_move(ch.move);
                auto sqname = [](int sq) {
                    char buf[3] = {static_cast<char>('a' + file_of(sq)), static_cast<char>('1' + rank_of(sq)), 0};
                    return std::string(buf);
                };
                std::string uci = sqname(m.from) + sqname(m.to);
                if (m.promo >= 0) uci += "?nbrq"[m.promo];
                g.moves.push_back(uci);

                s.apply_root_move(ch.move, ch.child_node);
                ++g.ply;
                return false;
            };

            int wait_ms = 0;
            for (;;) {
                ready.clear();
            
                // Check which batchers this worker currently has in-flight tasks for
                bool waiting_a = false, waiting_b = false;
                for (const auto& g : slots) {
                    if (!g.search || g.exhausted || g.inflight == 0) continue;
                    const bool a_moves = ((g.search->root_position().stm == WHITE) == g.a_is_white);
                    if (a_moves) waiting_a = true;
                    else waiting_b = true;
                }
                
                // 1. Wait on A only if we have tasks for A
                size_t got = batch_a.drain_wait(t, ready, waiting_a ? wait_ms : 0);
                {
                    std::vector<EvalBatcher::Item> rb;
                    // 2. Wait on B ONLY if we have tasks for B AND we didn't already receive items from A
                    int wait_b = (waiting_b && got == 0) ? wait_ms : 0;
                    got += batch_b.drain_wait(t, rb, wait_b);
                    ready.insert(ready.end(), std::make_move_iterator(rb.begin()), std::make_move_iterator(rb.end()));
                }

                for (auto& item : ready) {
                    for (auto& g : slots) {
                        if (!g.search || g.exhausted) continue;
                        if (g.search.get() == item.search) { g.inflight -= 1; break; }
                    }
                    item.search->complete_task(std::move(item.task));
                }

                bool any_progress = got > 0;
                bool all_exhausted = true;

                for (auto& g : slots) {
                    if (g.exhausted) continue;
                    all_exhausted = false;

                    if (g.move_ready) {
                        bool ended = conclude(g);
                        g.move_ready = false;
                        any_progress = true;
                        if (ended) {
                            double pts = g.result == 1 ? 0.5
                                : ((g.result == 0) == g.a_is_white ? 1.0 : 0.0);
                            if (pts == 1.0) aw.fetch_add(1);
                            else if (pts == 0.5) ad.fetch_add(1);
                            else al.fetch_add(1);
                            g.finished = true;

                            if (!pgn_out.empty()) {
                                std::string white_name = g.a_is_white ? model_a : model_b;
                                std::string black_name = g.a_is_white ? model_b : model_a;
                                std::string res_tag = (g.result == 0) ? "1-0"
                                                     : (g.result == 1) ? "1/2-1/2" : "0-1";
                            
                                std::string game_text;
                                game_text += "[Event \"Engine Match\"]\n";
                                game_text += "[Round \"" + std::to_string(g.game_id + 1) + "\"]\n";
                                game_text += "[White \"" + white_name + "\"]\n";
                                game_text += "[Black \"" + black_name + "\"]\n";
                                game_text += "[Result \"" + res_tag + "\"]\n\n";
                                for (size_t i = 0; i < g.moves.size(); ++i) {
                                    if (i % 2 == 0) game_text += std::to_string(i / 2 + 1) + ". ";
                                    game_text += g.moves[i] + " ";
                                }
                                game_text += res_tag + "\n\n";
                            
                                std::lock_guard<std::mutex> lk(pgn_mu);
                                std::ofstream f(pgn_out, std::ios::app);
                                if (f.is_open()) f << game_text;
                            }
                        }
                        continue;
                    }
                    {
                        const Tree::Node& rn = g.search->tree().node(g.search->root());
                        if (rn.state == Tree::ST_TERMINAL ||
                            (g.inflight == 0 && g.search->root_visits() >= visits)) {
                            g.move_ready = true;
                            continue;
                        }
                    }
                    if (static_cast<int>(g.search->root_visits() + g.inflight) < visits + 32) {
                        std::vector<EvalTask> ts;
                        ts.reserve(32);
                        g.search->gather_round(32, ts);
                        if (!ts.empty()) {
                            const bool a_moves =
                                ((g.search->root_position().stm == WHITE) ==
                                 g.a_is_white);
                            EvalBatcher& B = a_moves ? batch_a : batch_b;
                            g.inflight += ts.size();
                            for (auto& tk : ts) B.submit(t, g.search.get(), std::move(tk));
                            any_progress = true;
                        }
                    }
                }

                for (auto& g : slots) {
                    if (g.finished && !g.exhausted) {
                        int done = finished_games.fetch_add(1) + 1;
                        fprintf(stderr,
                                "{\"type\":\"matchprog\",\"done\":%d,\"total\":%d}\n",
                                done, games);
                        try_claim(g);
                        any_progress = true;
                    }
                }

                if (all_exhausted) break;
                wait_ms = any_progress ? 0 : 5;
            }
        });
    }
    for (auto& th : workers) th.join();
    batch_a.stop();
    batch_b.stop();

    int w = aw.load(), d = ad.load(), l = al.load();
    double score_a = w + 0.5 * d;
    double p = score_a / std::max(games, 1);
    
    double p_clamped = std::max(0.5 / std::max(games, 1), 
                                std::min(1.0 - 0.5 / std::max(games, 1), p));
    double elo = -400.0 * std::log10(1.0 / p_clamped - 1.0);
    double se = 400.0 / std::log(10.0) * std::sqrt(p_clamped * (1.0 - p_clamped) / std::max(games, 1));

    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"type\":\"match\",\"games\":%d,\"wins\":%d,\"draws\":%d,\"losses\":%d,"
             "\"score\":%.3f,\"elo_diff\":%.1f,\"elo_se\":%.1f}",
             games, w, d, l, p, elo, se);
    return std::string(buf);
}

int run_bench(const std::string& model_path, int seconds, int batch_size, bool prefer_gpu) {
    attacks::init();
    NNEvaluator nn;
    std::string err;
    if (!nn.load(model_path, batch_size, prefer_gpu, &err)) {
        fprintf(stderr, "FATAL %s\n", err.c_str());
        return 1;
    }
    std::vector<uint64_t> planes(static_cast<size_t>(batch_size) * NUM_PLANES, 0xDEADBEEFCAFEBABEULL);
    std::vector<float> scalars(static_cast<size_t>(batch_size) * SCALAR_COUNT, 0.f);
    std::vector<float> p, wv, m;
    nn.evaluate(planes.data(), scalars.data(), batch_size, p, wv, m);
    auto t0 = std::chrono::steady_clock::now();
    long long iters = 0;
    for (;;) {
        nn.evaluate(planes.data(), scalars.data(), batch_size, p, wv, m);
        ++iters;
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (el >= seconds) {
            fprintf(stdout, "{\"type\":\"bench\",\"eps\":%.0f,\"batch\":%d,\"gpu\":%s}\n",
                    iters * batch_size / el, batch_size, nn.is_gpu() ? "true" : "false");
            break;
        }
    }
    return 0;
}

}  // namespace chess