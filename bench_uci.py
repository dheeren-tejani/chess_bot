#!/usr/bin/env python3
"""Benchmark engine UCI: fixed-visit latency across positions, statistical.

Usage:
    python -u bench_uci.py --engine engine/build/engine \
        --model models/champion.onnx --visits 1000 --repeats 5 \
        --out bench_after.json

Compare two runs:
    python -u bench_uci.py --compare bench_before.json bench_after.json
"""
import argparse, json, statistics, subprocess, sys, threading, time, queue
from pathlib import Path

POSITIONS = [
    ("startpos",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("open_mid",   "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 6 6"),
    ("sharp",      "r4rk1/1pp1qppp/p1np1n2/2b1p1b1/2B1P1B1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("endgame",    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("promotion",  "8/P7/8/8/8/8/8/k1K5 w - - 0 1"),
]

READ_TIMEOUT_S = 120.0   # per bestmove. Hard kill above this.


class UciEngine:
    def __init__(self, engine_bin, model_path, visits):
        self.proc = subprocess.Popen(
            [engine_bin, "uci", "--model", model_path, "--visits", str(visits)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)

        # Drain stderr in background and echo with a prefix.
        def pump_err():
            for line in self.proc.stderr:
                sys.stderr.write(f"[engine] {line.rstrip()}\n")
                sys.stderr.flush()
        threading.Thread(target=pump_err, daemon=True).start()

        # ONE persistent reader for stdout. Never spawn per-call readers.
        self._stdout_q = queue.Queue()
        def pump_out():
            for line in self.proc.stdout:
                self._stdout_q.put(line.rstrip())
            self._stdout_q.put(None)   # EOF sentinel
        threading.Thread(target=pump_out, daemon=True).start()

    def _send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _read_until(self, prefix, timeout_s):
        deadline = time.perf_counter() + timeout_s
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"no {prefix!r} within {timeout_s}s")
            try:
                line = self._stdout_q.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"no {prefix!r} within {timeout_s}s")
            if line is None:
                raise RuntimeError(f"engine died waiting for {prefix!r}")
            if line.startswith(prefix):
                return line

    def uci_init(self):
        print("  [init] sending uci...", flush=True)
        self._send("uci");    self._read_until("uciok", 30)
        print("  [init] uciok. isready...", flush=True)
        self._send("isready"); self._read_until("readyok", 30)
        print("  [init] readyok. ucinewgame.", flush=True)
        self._send("ucinewgame")

    def think(self, fen, visits):
        self._send(f"position fen {fen}")
        t0 = time.perf_counter()
        self._send(f"go nodes {visits}")
        line = self._read_until("bestmove", READ_TIMEOUT_S)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        mv = line.split()[1] if len(line.split()) > 1 else "0000"
        return dt_ms, mv

    def quit(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()

def run(engine_bin, model_path, visits, repeats, warmup=1):
    print(f"[bench] launching {engine_bin} --model {model_path} --visits {visits}",
          flush=True)
    eng = UciEngine(engine_bin, model_path, visits)
    eng.uci_init()

    print(f"[bench] warmup ({warmup} pass over {len(POSITIONS)} positions)", flush=True)
    for w in range(warmup):
        for name, fen in POSITIONS:
            dt, mv = eng.think(fen, visits)
            print(f"  [warmup {w+1}] {name:<10} {dt:8.1f}ms -> {mv}", flush=True)

    results = {}
    for name, fen in POSITIONS:
        print(f"[bench] position {name}", flush=True)
        times, moves = [], []
        for r in range(repeats):
            dt, mv = eng.think(fen, visits)
            times.append(dt); moves.append(mv)
            print(f"  rep {r+1}/{repeats}: {dt:8.1f}ms -> {mv}", flush=True)
        results[name] = {
            "times_ms": times, "moves": moves,
            "median_ms": statistics.median(times),
            "min_ms": min(times), "max_ms": max(times),
            "mean_ms": statistics.mean(times),
        }
    eng.quit()
    return results


def report(tag, results, visits):
    print(f"\n=== {tag}  (visits={visits}) ===")
    print(f"{'position':<12} {'median':>10} {'min':>10} {'max':>10} {'nps_med':>10}  bestmove")
    all_med = []
    for name, r in results.items():
        nps = visits / (r["median_ms"] / 1000.0) if r["median_ms"] > 0 else 0
        all_med.append(r["median_ms"])
        print(f"{name:<12} {r['median_ms']:>9.1f}ms "
              f"{r['min_ms']:>9.1f}ms {r['max_ms']:>9.1f}ms "
              f"{nps:>9.0f}  {r['moves'][0]}")
    geo = statistics.geometric_mean(all_med) if all_med else 0
    print(f"{'GEOMEAN':<12} {geo:>9.1f}ms")
    return geo


def compare(path_a, path_b):
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    print(f"\n=== A: {path_a}")
    print(f"=== B: {path_b}")
    print(f"{'position':<12} {'A_med':>10} {'B_med':>10} {'speedup':>10}  "
          f"{'A_move':>8}  {'B_move':>8}  agree")
    geos = []
    for name in a["results"]:
        ra, rb = a["results"][name], b["results"][name]
        sp = ra["median_ms"] / rb["median_ms"] if rb["median_ms"] > 0 else 0
        geos.append(sp)
        agree = "yes" if ra["moves"][0] == rb["moves"][0] else "NO"
        print(f"{name:<12} {ra['median_ms']:>9.1f} {rb['median_ms']:>9.1f} "
              f"{sp:>9.2f}x  {ra['moves'][0]:>8}  {rb['moves'][0]:>8}  {agree}")
    geo = statistics.geometric_mean(geos) if geos else 0
    print(f"{'GEOMEAN':<12} {'':>10} {'':>10} {geo:>9.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="engine/build/engine")
    ap.add_argument("--model", default="models/champion.onnx")
    ap.add_argument("--visits", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare); sys.exit(0)

    results = run(args.engine, args.model, args.visits, args.repeats)
    geo = report("benchmark", results, args.visits)
    if args.out:
        Path(args.out).write_text(json.dumps({
            "engine": args.engine, "model": args.model,
            "visits": args.visits, "repeats": args.repeats,
            "geomean_median_ms": geo, "results": results,
        }, indent=2))
        print(f"\nwrote {args.out}")