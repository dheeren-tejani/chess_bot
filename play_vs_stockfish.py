#!/usr/bin/env python3
import argparse
import math
import multiprocessing as mp
import subprocess
import sys
import chess

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

def get_bestmove(proc, go_cmd: str) -> str | None:
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
        [cfg["engine_bin"], "uci", "--model", cfg["model_path"], "--visits", str(cfg["visits"])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1
    )
    sf = subprocess.Popen(
        [cfg["stockfish_path"]],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1
    )

    send(bot, "uci")
    read_until(bot, "uciok")
    send(bot, "isready")
    read_until(bot, "readyok")

    send(sf, "uci")
    read_until(sf, "uciok")
    send(sf, f"setoption name Skill Level value {cfg['skill_level']}")
    send(sf, "isready")
    read_until(sf, "readyok")

    send(bot, "ucinewgame")
    send(sf, "ucinewgame")

    board = chess.Board()
    moves = []

    while not board.is_game_over(claim_draw=True) and len(moves) < 300:
        pos_str = "position startpos" + ((" moves " + " ".join(moves)) if moves else "")
        is_bot = (board.turn == chess.WHITE) == bot_is_white

        if is_bot:
            send(bot, pos_str)
            mv = get_bestmove(bot, f"go nodes {cfg['visits']}")
        else:
            send(sf, pos_str)
            mv = get_bestmove(sf, f"go movetime {cfg['movetime']}")

        if not mv or mv in ("0000", "(none)"):
            break

        try:
            m = chess.Move.from_uci(mv)
            if m not in board.legal_moves:
                break
            board.push(m)
            moves.append(mv)
        except Exception:
            break

    for p in (bot, sf):
        try:
            send(p, "quit")
            p.wait(timeout=0.2)
        except Exception:
            p.kill()

    res = board.result(claim_draw=True)
    if res == "1-0":
        score = 1.0 if bot_is_white else 0.0
    elif res == "0-1":
        score = 0.0 if bot_is_white else 1.0
    else:
        score = 0.5

    tag = "WIN " if score == 1.0 else ("DRAW" if score == 0.5 else "LOSS")
    print(f"Game {game_idx + 1:3d} | Bot ({('White' if bot_is_white else 'Black'):5s}) | {tag} | Plies: {len(moves):3d} | [{res}]")
    return score

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
    args = parser.parse_args()

    cfg = {
        "engine_bin": args.engine,
        "model_path": args.model,
        "stockfish_path": args.stockfish,
        "visits": args.visits,
        "skill_level": args.skill,
        "movetime": args.movetime,
    }

    tasks = [(i, cfg) for i in range(args.games)]
    print(f"Starting match: {args.games} games, {args.workers} workers, Bot visits={args.visits}, SF Skill={args.skill}")

    with mp.Pool(processes=args.workers) as pool:
        scores = pool.map(play_one_game, tasks)

    wins = scores.count(1.0)
    draws = scores.count(0.5)
    losses = scores.count(0.0)
    tot = len(scores)
    pts = wins + 0.5 * draws
    pct = pts / tot

    clamped = max(0.5 / tot, min(1.0 - 0.5 / tot, pct))
    elo_diff = -400.0 * math.log10(1.0 / clamped - 1.0)

    print("\n" + "=" * 45)
    print(f"Final: +{wins} ={draws} -{losses} ({pct * 100:.1f}%)")
    print(f"Estimated Elo diff vs SF Level {args.skill}: {elo_diff:+.1f}")
    print("=" * 45)

if __name__ == "__main__":
    main()