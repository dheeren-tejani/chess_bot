#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "position.h"
#include "policy_map.h"
#include "encoder.h"
#include "selfplay.h"

using namespace chess;

static void cmd_perft(int argc, char** argv) {
    // perft "<fen>" d1 [d2 ...]
    attacks::init();
    Position pos;
    pos.set_from_fen(argv[2]);
    for (int i = 3; i < argc; ++i) {
        int d = std::stoi(argv[i]);
        uint64_t n = perft(pos, d);
        printf("perft %d = %llu\n", d, (unsigned long long)n);
    }
}

static void cmd_perft_split(int argc, char** argv) {
    // perft-split "<fen>" depth -> "e2e4 <count>" per legal root move
    (void)argc;
    attacks::init();
    Position pos;
    pos.set_from_fen(argv[2]);
    int d = std::stoi(argv[3]);
    std::vector<Move> moves;
    gen_legal_moves(pos, moves);
    auto sqname = [](int s) {
        char buf[3] = {static_cast<char>('a' + file_of(s)), static_cast<char>('1' + rank_of(s)), 0};
        return std::string(buf);
    };
    for (const Move& m : moves) {
        Position np = make_move(pos, m);
        std::string promo;
        if (m.promo >= 0) promo = std::string(1, "?nbrq"[m.promo]);
        printf("%s%s%s %llu\n", sqname(m.from).c_str(), sqname(m.to).c_str(), promo.c_str(),
               (unsigned long long)(d <= 1 ? 1 : perft(np, d - 1)));
    }
}

static void cmd_policy_index(int argc, char** argv) {
    // policy-index "<fen>"  ->  one line per legal move: "e2e4 1234"
    (void)argc;
    attacks::init();
    Position pos;
    pos.set_from_fen(argv[2]);
    std::vector<Move> moves;
    gen_legal_moves(pos, moves);
    Color mover = static_cast<Color>(pos.stm);
    auto sqname = [](int s) {
        char buf[3] = {static_cast<char>('a' + file_of(s)), static_cast<char>('1' + rank_of(s)), 0};
        return std::string(buf);
    };
    for (const Move& m : moves) {
        int idx = move_policy_index(m, mover);
        std::string promo;
        if (m.promo >= 0) promo = std::string(1, "?nbrq"[m.promo]);
        printf("%s%s%s %d\n", sqname(m.from).c_str(), sqname(m.to).c_str(), promo.c_str(), idx);
    }
}

static void cmd_encode(int argc, char** argv) {
    // encode "<fen>" -> prints 112 hex plane words then 4 scalars
    (void)argc;
    attacks::init();
    Position pos;
    pos.set_from_fen(argv[2]);
    EncodedPosition e = encode_position(pos, {pos});
    for (uint64_t w : e.planes) printf("%016lx\n", (unsigned long)w);
    for (float f : e.scalars) printf("%.6f ", f);
    printf("\n%d\n", material_diff_stm(pos));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: engine <command>\n"
                "  perft \"<fen>\" d...\n"
                "  perft-split \"<fen>\" depth\n"
                "  policy-index \"<fen>\"\n"
                "  encode \"<fen>\"\n"
                "  selfplay --model M --out D [--games N] [--threads T] [--batch B]\n"
                "           [--seed S] [--fast-visits V] [--full-visits V] [--fast-prob P]\n"
                "           [--temp-plies N] [--prior-plies N] [--max-plies N]\n"
                "           [--leaves-per-round N] [--shard-games G] [--cpu]\n"
                "           [--shard-start N] [--adjudicate-pawns P]\n"
                "           [--resign-threshold Q] [--resign-min-ply N]\n"
                "           [--resign-consecutive N] [--resign-continue F]\n"
                "  match --model-a A --model-b B [--games N] [--visits V] [--batch B]\n"
                "        [--threads T] [--games-per-worker G] [--cpu]\n"
                "  bench --model M [--seconds S] [--batch B] [--cpu]\n");
        return 2;
    }

    if (!strcmp(argv[1], "perft") && argc >= 4) { cmd_perft(argc, argv); return 0; }
    if (!strcmp(argv[1], "perft-split") && argc >= 4) { cmd_perft_split(argc, argv); return 0; }
    if (!strcmp(argv[1], "policy-index") && argc >= 3) { cmd_policy_index(argc, argv); return 0; }
    if (!strcmp(argv[1], "encode") && argc >= 3) { cmd_encode(argc, argv); return 0; }

    if (!strcmp(argv[1], "selfplay")) {
        std::string model, out = ".";
        int games = 1000, threads = 8, batch = 256, seed = 1, shard_games = 64;
        int games_per_worker = 4;
        int shard_start = 0;
        SelfPlayConfig cfg;
        bool cpu = false;
        for (int i = 2; i < argc; ++i) {
            auto nexti = [&]() { return std::stoi(argv[++i]); };
            auto nextf = [&]() { return static_cast<float>(std::stof(argv[++i])); };
            auto nexts = [&]() { return std::string(argv[++i]); };
            if (!strcmp(argv[i], "--model")) model = nexts();
            else if (!strcmp(argv[i], "--out")) out = nexts();
            else if (!strcmp(argv[i], "--games")) games = nexti();
            else if (!strcmp(argv[i], "--threads")) threads = nexti();
            else if (!strcmp(argv[i], "--batch")) cfg.batch_size = batch = nexti();
            else if (!strcmp(argv[i], "--seed")) seed = nexti();
            else if (!strcmp(argv[i], "--shard-games")) shard_games = nexti();
            else if (!strcmp(argv[i], "--fast-visits")) cfg.fast_visits = nexti();
            else if (!strcmp(argv[i], "--full-visits")) cfg.full_visits = nexti();
            else if (!strcmp(argv[i], "--fast-prob")) cfg.fast_prob = nextf();
            else if (!strcmp(argv[i], "--temp-plies")) cfg.temperature_plies = cfg.mcts.temperature_plies = nexti();
            else if (!strcmp(argv[i], "--prior-plies")) cfg.prior_plies = nexti();
            else if (!strcmp(argv[i], "--max-plies")) cfg.max_plies = nexti();
            else if (!strcmp(argv[i], "--leaves-per-round")) cfg.leaves_per_round = nexti();
            else if (!strcmp(argv[i], "--games-per-worker")) games_per_worker = nexti();
            else if (!strcmp(argv[i], "--shard-start")) shard_start = nexti();
            else if (!strcmp(argv[i], "--adjudicate-pawns")) cfg.adjudicate_material_pawns = nextf();
            else if (!strcmp(argv[i], "--value-mat-alpha")) cfg.value_material_alpha = nextf();
            else if (!strcmp(argv[i], "--contempt")) cfg.contempt = nextf();
            else if (!strcmp(argv[i], "--resign-threshold")) { cfg.resign_enabled = true; cfg.resign_threshold = nextf(); }
            else if (!strcmp(argv[i], "--resign-min-ply")) { cfg.resign_enabled = true; cfg.resign_min_ply = nexti(); }
            else if (!strcmp(argv[i], "--resign-consecutive")) { cfg.resign_enabled = true; cfg.resign_consecutive = nexti(); }
            else if (!strcmp(argv[i], "--resign-continue")) { cfg.resign_enabled = true; cfg.resign_continue_frac = nextf(); }
            else if (!strcmp(argv[i], "--no-dirichlet")) cfg.mcts.root_dirichlet = false;
            else if (!strcmp(argv[i], "--no-twofold")) cfg.mcts.use_twofold_draw = false;
            else if (!strcmp(argv[i], "--cpu")) cpu = true;
        }
        if (model.empty()) { fprintf(stderr, "--model required\n"); return 2; }
        // keep SelfPlayConfig and its embedded MCTSConfig in sync regardless of
        // whether the flags were passed (mirrors the defaults on both structs)
        cfg.mcts.temperature_plies = cfg.temperature_plies;
        cfg.mcts.prior_plies = cfg.prior_plies;
        return run_selfplay(model, out, games, threads, cfg,
                            static_cast<uint64_t>(seed), shard_games, !cpu,
                            games_per_worker, shard_start);
    }

    if (!strcmp(argv[1], "match")) {
        std::string a, b;
        int games = 60, visits = 320, batch = 128, threads = 4, gpw = 3;
        bool cpu = false;
        for (int i = 2; i < argc; ++i) {
            auto nexti = [&]() { return std::stoi(argv[++i]); };
            auto nexts = [&]() { return std::string(argv[++i]); };
            if (!strcmp(argv[i], "--model-a")) a = nexts();
            else if (!strcmp(argv[i], "--model-b")) b = nexts();
            else if (!strcmp(argv[i], "--games")) games = nexti();
            else if (!strcmp(argv[i], "--visits")) visits = nexti();
            else if (!strcmp(argv[i], "--batch")) batch = nexti();
            else if (!strcmp(argv[i], "--threads")) threads = nexti();
            else if (!strcmp(argv[i], "--games-per-worker")) gpw = nexti();
            else if (!strcmp(argv[i], "--cpu")) cpu = true;
        }
        if (a.empty() || b.empty()) { fprintf(stderr, "--model-a/--model-b required\n"); return 2; }
        printf("%s\n", run_match(a, b, games, visits, batch, !cpu, threads, gpw).c_str());
        return 0;
    }

    if (!strcmp(argv[1], "bench")) {
        std::string model;
        int seconds = 10, batch = 256;
        bool cpu = false;
        for (int i = 2; i < argc; ++i) {
            std::string k = argv[i];
            auto nexti = [&]() { return std::stoi(argv[++i]); };
            auto nexts = [&]() { return std::string(argv[++i]); };
            if (k == "--model") model = nexts();
            else if (k == "--seconds") seconds = nexti();
            else if (k == "--batch") batch = nexti();
            else if (k == "--cpu") cpu = true;
        }
        if (model.empty()) { fprintf(stderr, "--model required\n"); return 2; }
        return run_bench(model, seconds, batch, !cpu);
    }

    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 2;
}
