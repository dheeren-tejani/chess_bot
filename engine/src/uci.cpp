#include "uci.h"
#include "selfplay.h"   // for EvalBatcher only — no behaviour in selfplay.cpp is touched

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mcts.h"
#include "nn_client.h"

namespace chess {

namespace {

constexpr const char* START_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

std::string move_to_uci(const Move& m) {
    std::string s;
    s += static_cast<char>('a' + file_of(m.from));
    s += static_cast<char>('1' + rank_of(m.from));
    s += static_cast<char>('a' + file_of(m.to));
    s += static_cast<char>('1' + rank_of(m.to));
    if (m.promo >= 0) s += "?nbrq"[m.promo];   // KNIGHT=1,BISHOP=2,ROOK=3,QUEEN=4
    return s;
}

bool parse_uci_move(const Position& pos, const std::string& u, Move& out) {
    std::vector<Move> legal;
    gen_legal_moves(pos, legal);
    for (const Move& m : legal)
        if (move_to_uci(m) == u) { out = m; return true; }
    return false;
}

int cp_from_q(float q) {   // [-1,1] -> centipawns (logistic, like most MCTS engines)
    q = std::max(-0.995f, std::min(0.995f, q));
    double p = (q + 1.0) * 0.5;
    return static_cast<int>(400.0 * std::log10(p / (1.0 - p)));
}

}  // namespace

class UciPlayer {
public:
    static constexpr int NUM_WORKERS = 32;
    static constexpr int BATCH_SIZE  = 32;   // keep == NUM_WORKERS

    bool load(const std::string& model, bool prefer_gpu) {
        std::string err;
        if (!nn_.load(model, BATCH_SIZE, prefer_gpu, &err)) {
            std::cout << "info string model load failed: " << err << std::endl;
            return false;
        }
        std::cout << "info string model loaded (gpu="
                  << (nn_.is_gpu() ? "yes" : "no")
                  << ", batch=" << BATCH_SIZE
                  << ", workers=" << NUM_WORKERS << ")\n";

        cfg_.root_dirichlet = false;
        cfg_.temperature_plies = 0;
        cfg_.prior_plies = 0;
        cfg_.use_twofold_draw = true;

        // contempt=0.5, value_material_alpha=0 -> identical value math to the
        // original synchronous UCI eval().
        batcher_ = std::make_unique<EvalBatcher>(nn_, BATCH_SIZE, 0.0f, 0.5f);
        batcher_->set_mailboxes(NUM_WORKERS);
        batcher_->start();   // persistent across `go` commands
        return true;
    }

    ~UciPlayer() {
        if (batcher_) batcher_->stop();
    }

    uint16_t think(std::vector<Position> hist, int max_visits, double time_s) {
        stop_.store(false);
        const auto t0 = std::chrono::steady_clock::now();
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
        };

        // Sanity: any legal move at all? Also catches a root that is already
        // mate/stalemate before we spawn any workers.
        {
            Search probe(cfg_, std::vector<Position>(hist), &rng_);
            std::vector<Move> legal;
            gen_legal_moves(probe.root_position(), legal);
            if (legal.empty()) return 0;
        }

        // K independent searches of the same root, each producing one leaf
        // per round; the EvalBatcher aggregates across all K into real batches.
        std::vector<std::unique_ptr<Search>> searches;
        searches.reserve(NUM_WORKERS);
        for (int i = 0; i < NUM_WORKERS; ++i)
            searches.push_back(std::make_unique<Search>(
                cfg_, std::vector<Position>(hist), &rng_));

        std::atomic<int> total_visits{0};
        std::vector<std::thread> workers;
        workers.reserve(NUM_WORKERS);

        for (int i = 0; i < NUM_WORKERS; ++i) {
            workers.emplace_back([&, i] {
                Search& s = *searches[i];
                std::vector<EvalTask> ts;
                std::vector<EvalBatcher::Item> ready;
                int in_flight = 0;

                while (total_visits.load(std::memory_order_relaxed) < max_visits
                       && !stop_.load() && elapsed() < time_s) {

                    // (1) ALWAYS drain first — even if the previous gather
                    // produced nothing. The batcher may already have delivered
                    // a result into our mailbox.
                    ready.clear();
                    const int wait_ms = (in_flight > 0) ? 5 : 0;
                    batcher_->drain_wait(i, ready, wait_ms);
                    for (auto& item : ready) {
                        item.search->complete_task(std::move(item.task));
                        total_visits.fetch_add(1, std::memory_order_relaxed);
                        --in_flight;
                    }

                    // (2) Try to produce one new leaf.
                    ts.clear();
                    s.gather_round(1, ts);
                    if (!ts.empty()) {
                        for (auto& t : ts) {
                            batcher_->submit(i, &s, std::move(t));
                            ++in_flight;
                        }
                        continue;
                    }

                    // (3) Nothing new to gather this round.
                    if (in_flight > 0) {
                        // We just blocked on drain above; loop and block again.
                        continue;
                    }
                    if (s.tree().node(s.root()).state == Tree::ST_TERMINAL) {
                        break;   // root itself is mate/stalemate -> nothing left to search
                    }
                    // Transient: no leaf, no in-flight, but not terminal.
                    // Another worker must unblock us soon — tiny yield.
                    std::this_thread::sleep_for(
                        std::chrono::microseconds(200));
                }

                // Flush any last in-flight task BEFORE `s` is destroyed.
                // Without this, the batcher would later deliver into mailbox
                // `i`, and the next think() would call complete_task() on a
                // freed Search (use-after-free).
                const auto flush_deadline =
                    std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (in_flight > 0
                       && std::chrono::steady_clock::now() < flush_deadline) {
                    ready.clear();
                    batcher_->drain_wait(i, ready, 50);
                    for (auto& item : ready) {
                        item.search->complete_task(std::move(item.task));
                        total_visits.fetch_add(1, std::memory_order_relaxed);
                        --in_flight;
                    }
                }
            });
        }
        for (auto& w : workers) w.join();

        // Merge root visit/value stats across the K searches, keyed by packed move.
        std::unordered_map<uint16_t, std::pair<uint32_t, float>> agg;
        for (auto& sp : searches) {
            const Tree::Node& n = sp->tree().node(sp->root());
            const Tree::Edge* es = sp->tree().edges(sp->root());
            for (int j = 0; j < n.num; ++j) {
                auto& e = agg[es[j].move];
                e.first  += es[j].visits;
                e.second += es[j].value_sum;
            }
        }
        if (agg.empty()) return 0;

        uint16_t best_move = 0;
        uint32_t best_visits = 0, tot_visits = 0;
        double   tot_val = 0.0;
        for (auto& [mv, vv] : agg) {
            tot_visits += vv.first;
            tot_val    += vv.second;
            if (vv.first > best_visits) {
                best_visits = vv.first;
                best_move = mv;
            }
        }

        if (verbose_) {
            double q = tot_visits ? tot_val / tot_visits : 0.0;
            std::cout << "info depth " << (tot_visits / 32 + 1)
                      << " score cp " << cp_from_q(static_cast<float>(q))
                      << " nodes "   << tot_visits
                      << " nps "     << (elapsed() > 0
                                        ? int(tot_visits / elapsed()) : 0)
                      << " time "    << int(elapsed() * 1000)
                      << " pv "      << move_to_uci(unpack_move(best_move))
                      << "\n";
        }
        return best_move;
    }

    std::atomic<bool> stop_{false};
    bool verbose_ = true;

private:
    NNEvaluator nn_;
    MCTSConfig cfg_;
    std::unique_ptr<EvalBatcher> batcher_;
    std::mt19937_64 rng_{0x5EED1234};
};

int run_uci(const std::string& model_path, int default_visits, bool prefer_gpu) {
    attacks::init();
    UciPlayer player;
    if (!player.load(model_path, prefer_gpu)) return 1;

    std::vector<Position> hist(1);
    hist[0].set_from_fen(START_FEN);

    std::thread worker;
    auto join_worker = [&] {
        player.stop_.store(true);
        if (worker.joinable()) worker.join();
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream is(line);
        std::string tok;
        is >> tok;

        if (tok == "uci") {
            std::cout << "id name TabulaRasaBot\n"
                         "id author Dheer\n"
                         "option name Visits type spin default " << default_visits
                      << " min 16 max 1000000\n"
                         "uciok" << std::endl;
        } else if (tok == "isready") {
            std::cout << "readyok" << std::endl;
        } else if (tok == "ucinewgame") {
            join_worker();
            hist.assign(1, Position{});
            hist[0].set_from_fen(START_FEN);
        } else if (tok == "setoption") {
            std::string word, name, v; int val = 0;
            is >> word >> name >> v >> val;              // "name Visits value N"
            if (name == "Visits" && val > 0) default_visits = val;
        } else if (tok == "position") {
            join_worker();
            std::string what;
            is >> what;
            if (what == "startpos") {
                hist.assign(1, Position{});
                hist[0].set_from_fen(START_FEN);
                std::string kw;
                if (is >> kw && kw == "moves") {
                    std::string mv;
                    while (is >> mv) {
                        Move m;
                        if (parse_uci_move(hist.back(), mv, m))
                            hist.push_back(make_move(hist.back(), m));
                    }
                }
            } else if (what == "fen") {
                std::string fen, kw;
                for (int i = 0; i < 6; ++i) {            // up to 6 FEN fields
                    std::string part;
                    if (!(is >> part) || part == "moves") { kw = part; break; }
                    fen += part; fen += ' ';
                }
                hist.assign(1, Position{});
                hist[0].set_from_fen(fen);
                if (kw != "moves") is >> kw;
                if (kw == "moves") {
                    std::string mv;
                    while (is >> mv) {
                        Move m;
                        if (parse_uci_move(hist.back(), mv, m))
                            hist.push_back(make_move(hist.back(), m));
                        else
                            std::cout << "info string illegal move ignored: " << mv
                                      << std::endl;
                    }
                }
            }
        } else if (tok == "go") {
            join_worker();
            int nodes = -1, movetime = -1, wtime = -1, btime = -1,
                winc = 0, binc = 0, mtg = 0, depth = -1;
            bool infinite = false;
            std::string w;
            while (is >> w) {
                if (w == "nodes") is >> nodes;
                else if (w == "movetime") is >> movetime;
                else if (w == "wtime") is >> wtime;
                else if (w == "btime") is >> btime;
                else if (w == "winc") is >> winc;
                else if (w == "binc") is >> binc;
                else if (w == "movestogo") is >> mtg;
                else if (w == "depth") is >> depth;
                else if (w == "infinite") infinite = true;
                // searchmoves/ponder/mate: ignored gracefully
            }
            int visits = nodes > 0 ? nodes : (depth > 0 ? depth * 100 : default_visits);
            double tlimit = 1e9;
            if (movetime > 0) {
                tlimit = movetime / 1000.0 * 0.95;
            } else if (wtime >= 0 && btime >= 0) {
                int t = hist.back().stm == WHITE ? wtime : btime;
                int inc = hist.back().stm == WHITE ? winc : binc;
                double slice = t / (mtg > 0 ? mtg : 30.0) + 0.5 * inc;
                tlimit = std::min(slice, t * 0.5) / 1000.0;   // never burn >50%
                visits = 10000000;                            // time-capped instead
            }
            if (infinite) { tlimit = 1e9; visits = 100000000; }

            std::vector<Position> hcopy = hist;   // worker owns a copy
            worker = std::thread([&player, h = std::move(hcopy), visits, tlimit]() mutable {
                uint16_t best = player.think(std::move(h), visits, tlimit);
                std::cout << "bestmove "
                          << (best ? move_to_uci(unpack_move(best)) : "0000")
                          << std::endl;
            });
        } else if (tok == "stop") {
            join_worker();
        } else if (tok == "quit") {
            join_worker();
            break;
        }
    }
    join_worker();
    return 0;
}

}  // namespace chess