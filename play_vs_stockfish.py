#!/usr/bin/env python3
import argparse
import math
import multiprocessing as mp
import subprocess
import sys
import time

import chess

# Fair mode: bot is purely time-controlled. Huge visit ceiling makes the
# clock the binding constraint (this engine's UCI: `go movetime T nodes N`
# stops at whichever comes first).
BOT_VISIT_CEILING = 10_000_000


def send(proc, cmd: str):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_until(proc, prefix: str) -> str:
    while True:
        line = proc.stdout.readline()
        if not line:
            return ""
        line = line.strip()
        if line.startswith(prefix):
            return line


def get_bestmove(proc, go_cmd: str):
    send(proc, go_cmd)
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else None


def play_one_game(args_tuple):
    game_idx, cfg = args_tuple
    bot_is_white = (game_idx % 2 == 0)

    bot = subprocess.Popen(
        [cfg["engine_bin"], "uci", "--model", cfg["model_path"],
         "--visits", str(cfg["visits"])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    sf = subprocess.Popen(
        [cfg["stockfish_path"]],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )

    send(bot, "uci");     read_until(bot, "uciok")
    send(bot, "isready"); read_until(bot, "readyok")

    send(sf, "uci")
    elo_lo, elo_hi = 1320, 3190
    while True:
        line = sf.stdout.readline()
        if not line:
            break
        s = line.strip()
        if s.startswith("option name UCI_Elo"):
            p = s.split()
            if "min" in p: elo_lo = int(p[p.index("min") + 1])
            if "max" in p: elo_hi = int(p[p.index("max") + 1])
        if s.startswith("uciok"):
            break

    if cfg["sf_elo"] > 0:
        elo = min(max(cfg["sf_elo"], elo_lo), elo_hi)
        if elo != cfg["sf_elo"]:
            print(f"[game {game_idx+1}] --sf-elo {cfg['sf_elo']} outside SF range "
                  f"[{elo_lo},{elo_hi}] -> clamped to {elo}")
        send(sf, "setoption name UCI_LimitStrength value true")
        send(sf, f"setoption name UCI_Elo value {elo}")
    else:
        send(sf, f"setoption name Skill Level value {cfg['skill_level']}")
    send(sf, "setoption name Threads value 1")
    send(sf, "setoption name Hash value 64")
    send(sf, "isready"); read_until(sf, "readyok")

    send(bot, "ucinewgame")
    send(sf, "ucinewgame")

    if cfg["fair"]:
        bot_go = f"go movetime {int(round(cfg['movetime'] / 0.95))} nodes {BOT_VISIT_CEILING}"
        sf_go  = f"go movetime {cfg['movetime']}"
    else:
        bot_go = f"go nodes {cfg['visits']}"
        sf_go  = f"go movetime {cfg['movetime']}"

    board = chess.Board()
    moves = []
    bot_crashed = False
    bot_times = []            # seconds per bot move
    sf_times  = []            # seconds per SF move
    game_t0 = time.perf_counter()

    while not board.is_game_over(claim_draw=True) and len(moves) < 300:
        pos_str = "position startpos" + ((" moves " + " ".join(moves)) if moves else "")
        is_bot = (board.turn == chess.WHITE) == bot_is_white

        if is_bot:
            send(bot, pos_str)
            t0 = time.perf_counter()
            mv = get_bestmove(bot, bot_go)
            bot_times.append(time.perf_counter() - t0)
        else:
            send(sf, pos_str)
            t0 = time.perf_counter()
            mv = get_bestmove(sf, sf_go)
            sf_times.append(time.perf_counter() - t0)

        if not mv or mv in ("0000", "(none)"):
            if is_bot and bot.poll() is not None:
                bot_crashed = True
            break

        try:
            m = chess.Move.from_uci(mv)
            if m not in board.legal_moves:
                break
            board.push(m)
            moves.append(mv)
        except Exception:
            break

    game_elapsed = time.perf_counter() - game_t0

    for p in (bot, sf):
        try:
            send(p, "quit")
            p.wait(timeout=0.2)
        except Exception:
            p.kill()

    def avg_ms(xs):
        return (1000.0 * sum(xs) / len(xs)) if xs else 0.0

    if bot_crashed:
        print(f"Game {game_idx + 1:3d} | Bot ({('White' if bot_is_white else 'Black'):5s}) | "
              f"CRASH -> LOSS | {game_elapsed:6.1f}s "
              f"| bot {avg_ms(bot_times):6.0f}ms/mv  sf {avg_ms(sf_times):6.0f}ms/mv")
        return {"score": 0.0, "crashed": True, "game_s": game_elapsed,
                "plies": len(moves), "bot_times": bot_times, "sf_times": sf_times,
                "result": "*"}

    res = board.result(claim_draw=True)
    if res == "1-0":
        score = 1.0 if bot_is_white else 0.0
    elif res == "0-1":
        score = 0.0 if bot_is_white else 1.0
    else:
        score = 0.5

    tag = "WIN " if score == 1.0 else ("DRAW" if score == 0.5 else "LOSS")
    print(f"Game {game_idx + 1:3d} | Bot ({('White' if bot_is_white else 'Black'):5s}) | {tag} "
          f"| Plies: {len(moves):3d} | {game_elapsed:6.1f}s "
          f"| bot {avg_ms(bot_times):6.0f}ms/mv  sf {avg_ms(sf_times):6.0f}ms/mv "
          f"| [{res}]")

    return {"score": score, "crashed": False, "game_s": game_elapsed,
            "plies": len(moves), "bot_times": bot_times, "sf_times": sf_times,
            "result": res}


def main():
    parser = argparse.ArgumentParser(description="Parallel Bot vs Stockfish match runner")
    parser.add_argument("--engine", default="./engine/build/engine")
    parser.add_argument("--model", default="models/champion.onnx")
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--visits", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skill", type=int, default=0)
    parser.add_argument("--movetime", type=int, default=80)
    parser.add_argument("--sf-elo", type=int, default=0,
                        help="enable UCI_LimitStrength at this Elo (0 = use --skill instead)")
    parser.add_argument("--fair", action="store_true",
                        help="symmetric clock: BOTH engines get --movetime per move "
                             "(bot visits cap becomes non-binding)")
    args = parser.parse_args()

    if args.fair and args.movetime < 500:
        print("[warn] --fair with movetime < 500ms: SF's UCI_Elo calibration is "
              "defined for normal time controls; short clocks distort the ladder")

    cfg = {
        "engine_bin": args.engine,
        "model_path": args.model,
        "stockfish_path": args.stockfish,
        "visits": args.visits,
        "skill_level": args.skill,
        "movetime": args.movetime,
        "sf_elo": args.sf_elo,
        "fair": args.fair,
    }

    sf_mode = (f"UCI_Elo {args.sf_elo} (LimitStrength)" if args.sf_elo > 0
               else f"Skill {args.skill}")
    tc = (f"BOTH {args.movetime}ms/move" if args.fair
          else f"bot nodes {args.visits} vs SF {args.movetime}ms")
    print(f"Starting match: {args.games} games, {args.workers} workers | SF: {sf_mode} | TC: {tc}")

    wall_t0 = time.perf_counter()
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(play_one_game, [(i, cfg) for i in range(args.games)])
    wall_elapsed = time.perf_counter() - wall_t0

    scores = [r["score"] for r in results]
    wins = scores.count(1.0)
    draws = scores.count(0.5)
    losses = scores.count(0.0)
    tot = len(scores)
    pts = wins + 0.5 * draws
    pct = pts / tot

    clamped = max(0.5 / tot, min(1.0 - 0.5 / tot, pct))
    elo_diff = -400.0 * math.log10(1.0 / clamped - 1.0)
    se = 400.0 / (math.log(10) * math.sqrt(tot * clamped * (1.0 - clamped)))

    total_bot_s = sum(sum(r["bot_times"]) for r in results)
    total_sf_s  = sum(sum(r["sf_times"])  for r in results)
    n_bot_moves = sum(len(r["bot_times"]) for r in results)
    n_sf_moves  = sum(len(r["sf_times"])  for r in results)
    avg_bot_ms  = (1000.0 * total_bot_s / n_bot_moves) if n_bot_moves else 0.0
    avg_sf_ms   = (1000.0 * total_sf_s  / n_sf_moves)  if n_sf_moves  else 0.0
    avg_game_s  = sum(r["game_s"] for r in results) / tot
    avg_plies   = sum(r["plies"]  for r in results) / tot

    print("\n" + "=" * 55)
    print(f"Final: +{wins} ={draws} -{losses} ({pct * 100:.1f}%)")
    print(f"Elo vs SF [{sf_mode}] @ {tc}: {elo_diff:+.1f} ± {se:.0f} (1σ)")
    print("-" * 55)
    print(f"Wall time          : {wall_elapsed:7.1f}s total, {avg_game_s:5.1f}s/game")
    print(f"Bot think time     : {total_bot_s:7.1f}s total, {avg_bot_ms:5.0f}ms/move "
          f"({n_bot_moves} moves)")
    print(f"SF think time      : {total_sf_s:7.1f}s total, {avg_sf_ms:5.0f}ms/move "
          f"({n_sf_moves} moves)")
    print(f"Avg game length    : {avg_plies:5.1f} plies")
    print("=" * 55)


if __name__ == "__main__":
    main()