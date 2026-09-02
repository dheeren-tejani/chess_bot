#include "uci.h"                                                                                                                                           
                                                                                                                                                           
#include <algorithm>                                                                                                                                       
#include <atomic>                                                                                                                                          
#include <chrono>                                                                                                                                          
#include <cmath>                                                                                                                                           
#include <cstdio>                                                                                                                                          
#include <iostream>                                                                                                                                        
#include <sstream>                                                                                                                                         
#include <string>                                                                                                                                          
#include <thread>                                                                                                                                          
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
                                                                                                                                                           
// Synchronous (batch-1) driver around the existing Search.                                                                                                
class UciPlayer {                                                                                                                                          
public:                                                                                                                                                    
    bool load(const std::string& model, bool prefer_gpu) {                                                                                                 
        std::string err;                                                                                                                                   
        if (!nn_.load(model, 1 /*batch*/, prefer_gpu, &err)) {                                                                                             
            std::cout << "info string model load failed: " << err << std::endl;                                                                            
            return false;                                                                                                                                  
        }                                                                                                                                                  
        cfg_.root_dirichlet = false;    // no exploration noise when playing                                                                               
        cfg_.temperature_plies = 0;     // greedy                                                                                                          
        cfg_.prior_plies = 0;                                                                                                                              
        cfg_.use_twofold_draw = true;   // respect repetition (avoid 3-fold losses)                                                                        
        return true;                                                                                                                                       
    }                                                                                                                                                      
                                                                                                                                                           
    // Returns packed best move, or 0 if there are no legal moves.                                                                                         
    uint16_t think(std::vector<Position> hist, int max_visits, double time_s) {                                                                            
        stop_.store(false);                                                                                                                                
        auto t0 = std::chrono::steady_clock::now();                                                                                                        
        Search s(cfg_, std::move(hist), &rng_);                                                                                                            
                                                                                                                                                           
        {                                                                                                                                                  
            std::vector<Move> legal;                                                                                                                       
            gen_legal_moves(s.root_position(), legal);                                                                                                     
            if (legal.empty()) return 0;   // mated/stalemated already                                                                                     
        }                                                                                                                                                  
                                                                                                                                                           
        std::vector<EvalTask> tasks;
        auto elapsed = [&] {
            return std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
        };
        while (s.root_visits() < max_visits && !stop_.load() && elapsed() < time_s) {
            tasks.clear();
            // A single descent can end in a terminal backprop without producing
            // a task; keep descending until one produces work so the search
            // doesn't stop at the first terminal found mid-tree.
            while (tasks.empty() && s.root_visits() < max_visits && !stop_.load() &&
                   elapsed() < time_s)
                s.gather_round(1, tasks);
            if (tasks.empty()) break;   // nothing expandable left in the whole tree
            eval(tasks[0]);
            s.complete_task(std::move(tasks[0]));
            if (s.root_visits() % 64 == 0) print_info(s, elapsed());
        }                                                                                                                                                  
        auto ch = s.pick_move(rng_, 100000);  // ply >> any temperature -> greedy                                                                          
        return ch.move;                     // 0 only if root never expanded                                                                               
    }                                                                                                                                                      
                                                                                                                                                           
    std::atomic<bool> stop_{false};                                                                                                                        
                                                                                                                                                           
private:                                                                                                                                                   
    void eval(EvalTask& t) {   // mirrors EvalBatcher::fill_task, contempt=0.5                                                                             
        std::vector<float> policy, wdl, mat;                                                                                                               
        nn_.evaluate(t.enc.planes.data(), t.enc.scalars.data(), 1, policy, wdl, mat);                                                                         
                                                                                                                                                           
        const size_t K = t.moves.size();                                                                                                                   
        std::vector<int> idx(K);                                                                                                                           
        float maxl = -1e30f;                                                                                                                               
        for (size_t k = 0; k < K; ++k) {                                                                                                                   
            Move m = unpack_move(t.moves[k]);                                                                                                              
            idx[k] = move_policy_index(m, static_cast<Color>(t.stm));                                                                                      
            maxl = std::max(maxl, policy[idx[k]]);                                                                                                         
        }                                                                                                                                                  
        double sum = 0;                                                                                                                                    
        for (size_t k = 0; k < K; ++k) {                                                                                                                   
            t.priors[k] = std::exp(policy[idx[k]] - maxl);                                                                                                 
            sum += t.priors[k];                                                                                                                            
        }                                                                                                                                                  
        for (size_t k = 0; k < K; ++k) t.priors[k] /= static_cast<float>(sum);                                                                             
                                                                                                                                                           
        float mx = std::max({wdl[0], wdl[1], wdl[2]});                                                                                                     
        double e0 = std::exp(wdl[0] - mx), e1 = std::exp(wdl[1] - mx),                                                                                     
               e2 = std::exp(wdl[2] - mx);                                                                                                                 
        double ev = (e0 + 0.5 * e1) / (e0 + e1 + e2);   // neutral contempt                                                                                
        t.value = static_cast<float>(2.0 * ev - 1.0);   // engine [-1,1] convention                                                                        
    }                                                                                                                                                      
                                                                                                                                                           
    void print_info(const Search& s, double el) {                                                                                                          
        const Tree::Node& rn = s.tree().node(s.root());                                                                                                    
        const Tree::Edge* es = s.tree().edges(s.root());                                                                                                   
        int best = -1; uint32_t bv = 0;                                                                                                                    
        for (int i = 0; i < rn.num; ++i)                                                                                                                   
            if (es[i].visits > bv) { bv = es[i].visits; best = i; }                                                                                        
        if (best < 0) return;                                                                                                                              
        std::cout << "info depth " << (s.root_visits() / 32 + 1)   // MCTS: cosmetic                                                                       
                  << " score cp " << cp_from_q(s.root_q())                                                                                                 
                  << " nodes " << s.root_visits()                                                                                                          
                  << " nps " << (el > 0 ? int(s.root_visits() / el) : 0)                                                                                   
                  << " time " << int(el * 1000)                                                                                                            
                  << " pv " << move_to_uci(unpack_move(es[best].move)) << std::endl;                                                                       
    }                                                                                                                                                      
                                                                                                                                                           
    NNEvaluator nn_;                                                                                                                                       
    MCTSConfig cfg_;                                                                                                                                       
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