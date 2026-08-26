"""Orchestrator: endless self-play -> train -> gate -> promote loop.

Fully resumable: phase progress lives in orchestrator_state.json; training
resumes from checkpoints; shards are append-only.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import Config
from trainer.dataset import REC_DTYPE, RECORD_BYTES, ShardDataset, _f16_bits_to_float


def scan_shard_results(path: Path):
    """Return (n_positions, wins_white, draws, wins_black, material_mean).

    Fallback stats source; the engine's selfplay_summary JSON is preferred.
    Failures are REPORTED, never silently zeroed (that bug hid 12 dead
    iterations).
    """
    try:
        with gzip.open(path, "rb") as f:
            data = f.read()
        usable = (len(data) - 20) // RECORD_BYTES * RECORD_BYTES
        arr = __import__("numpy").frombuffer(data[20:20 + usable], dtype=REC_DTYPE)
        if len(arr) == 0:
            return 0, 0, 0, 0, 0.0
        bc = __import__("numpy").bincount(arr["result"], minlength=3)
        mat = _f16_bits_to_float(arr["material_f16"])
        return len(arr), int(bc[0]), int(bc[1]), int(bc[2]), float(mat.mean())
    except Exception as e:
        print(f"[warn] failed to scan shard {path.name}: {e}")
        return 0, 0, 0, 0, 0.0


# --------------------------------------------------------------------------
def gpu_env() -> dict:
    """LD_LIBRARY_PATH for ORT CUDA EP, sourced from torch's bundled libs."""
    import site
    sp = None
    try:
        sp = site.getsitepackages()[0]
    except Exception:
        pass
    if sp is None:
        return {}
    nvidia = Path(sp) / "nvidia"
    libs = [str(p) for p in nvidia.glob("*/lib")] if nvidia.exists() else []
    env = dict(os.environ)
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs) + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


class Orchestrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = Path.cwd()
        self.engine = self.root / "engine" / "build" / "engine"
        self.state_path = self.root / "orchestrator_state.json"
        self.model_dir = self.root / cfg.run.model_dir
        self.ckpt_dir = self.root / cfg.run.checkpoint_dir
        self.data_dir = self.root / cfg.run.data_dir
        for d in (self.model_dir, self.ckpt_dir, self.data_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.champion = self.model_dir / "champion.onnx"
        self.candidate = self.model_dir / "candidate.onnx"

        self.state = {"phase": "selfplay", "elo_history": [], "iterations": 0}
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text()))
            except Exception:
                pass

        self._stop = False
        signal.signal(signal.SIGINT, self._handle_sigint)

        # iteration-level wandb logging (training curves live in the trainer's run).
        # The loop-run id persists in state -> restarts REUSE the same dashboard
        # run instead of spawning zombie clones.
        self.wandb = None
        if cfg.wandb.enabled:
            wid = self.state.get("wandb_loop_id")
            try:
                import wandb
                kwargs = dict(project=cfg.wandb.project, entity=cfg.wandb.entity,
                              name=f"{cfg.run.name}-loop", job_type="orchestrator",
                              config=cfg.__dict__)
                if wid:
                    kwargs.update(id=wid, resume="allow")
                self.wandb = wandb.init(**kwargs)
                got = getattr(self.wandb, "id", None)
                if got and got != wid:
                    self.state["wandb_loop_id"] = got
                    self.save_state()
                # Loop metrics live on their own "iteration" x-axis so they can
                # never collide with the trainer's global step (the collision
                # that silently dropped most of the first run's curves).
                wandb.define_metric("iteration", hidden=True)
                for pref in ("sp_", "end_", "gate_", "anchor_", "champion_"):
                    wandb.define_metric(f"{pref}*", step_metric="iteration")
            except Exception as e:   # noqa: BLE001
                print(f"[warn] wandb unavailable ({e}); metrics still on console/state")
                self.wandb = None

    def log_iter(self, metrics: dict):
        metrics = {"iteration": self.state["iterations"], **metrics}
        if self.wandb is not None:
            try:
                self.wandb.log(metrics)
            except Exception:
                pass
        hist = self.state.setdefault("metrics_history", [])
        hist.append({k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in metrics.items()})
        del hist[:-50]   # keep last 50 iterations in state file

    def _handle_sigint(self, *_):
        print("\n[orchestrator] SIGINT: finishing current step, state saved.")
        self._stop = True

    def save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2))
        tmp.replace(self.state_path)

    # -- subprocess helpers --------------------------------------------------
    def run_engine(self, args: list[str], on_line=None, timeout=None) -> tuple[int, list[str]]:
        env = gpu_env()
        proc = subprocess.Popen([str(self.engine), *args], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
        out_lines: list[str] = []
        assert proc.stdout and proc.stderr

        import threading
        err_lines: list[str] = []

        def pump_err():
            for line in proc.stderr:
                err_lines.append(line.rstrip())
                self._last_stderr_tail = (getattr(self, "_last_stderr_tail", []) +
                                          [line.rstrip()])[-20:]
                if on_line:
                    on_line("stderr", line.rstrip())

        th = threading.Thread(target=pump_err, daemon=True)
        th.start()
        for line in proc.stdout:
            out_lines.append(line.rstrip())
            if on_line:
                on_line("stdout", line.rstrip())

        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
        th.join(timeout=5)
        return rc, out_lines + ["[E] " + l for l in err_lines[-40:]]

    # -- phases ---------------------------------------------------------------
    def export_candidate_if_needed(self):
        """Train a candidate ONNX from the latest checkpoint."""
        latest = self.ckpt_dir / "latest.pt"
        cmd = [sys.executable, str(self.root / "trainer" / "export_onnx.py"),
               "--out", str(self.candidate),
               "--blocks", str(self.cfg.net.blocks),
               "--channels", str(self.cfg.net.channels)]
        if latest.exists():
            cmd += ["--checkpoint", str(latest)]
        print("[export]", " ".join(cmd[2:]))
        r = subprocess.run(cmd, cwd=self.root)
        if r.returncode != 0:
            raise RuntimeError("onnx export failed")

    def phase_selfplay(self):
        c = self.cfg.selfplay
        it = self.state["iterations"]
        # CRITICAL: per-iteration directory - the engine restarts shard numbering
        # at zero every launch, so a shared dir would overwrite prior iterations'
        # data (this bug cost us 12 iterations of discarded games).
        out_dir = self.data_dir / f"iter_{it:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        known_shards = set(out_dir.glob("*.gz"))
        # Resume support: continue numbering AFTER shards an interrupted launch
        # already wrote, instead of overwriting them (their games would be lost
        # from the stats diff and duplicated on disk).
        args = [
            "selfplay",
            "--model", str(self.champion if self.champion.exists()
                           else self.candidate if self.candidate.exists() else self._bootstrap()),
            "--out", str(out_dir),
            "--games", str(c.games_per_phase),
            "--threads", str(c.threads),
            "--batch", str(c.batch_size),
            "--seed", str(c.seed + 1000 * it),
            "--fast-visits", str(c.fast_visits),
            "--full-visits", str(c.full_visits),
            "--fast-prob", str(c.fast_prob),
            "--leaves-per-round", str(c.leaves_per_round),
            "--temp-plies", str(c.temperature_plies),
            "--prior-plies", str(c.prior_plies),
            "--max-plies", str(c.max_plies),
            "--adjudicate-pawns", str(c.adjudicate_material_pawns),
            "--value-mat-alpha", str(c.value_material_alpha),
            "--contempt", str(c.contempt),
            "--shard-games", str(c.shard_games),
            "--games-per-worker", str(c.games_per_worker),
            "--shard-start", str(len(known_shards)),
        ]
        if not c.use_twofold_draw:
            args.append("--no-twofold")
        if c.resign_enabled:
            args += ["--resign-threshold", str(c.resign_threshold),
                     "--resign-min-ply", str(c.resign_min_ply),
                     "--resign-consecutive", str(c.resign_consecutive),
                     "--resign-continue", str(c.resign_continue_frac)]
        t0 = time.time()

        def on_line(stream, line):
            if '"type":"stats"' in line or '"type":"info"' in line:
                try:
                    obj = json.loads(line[line.index("{"):])
                    if obj.get("type") == "stats":
                        el = obj["elapsed_s"]
                        eta = (el / max(obj["games"], 1)) * max(
                            c.games_per_phase - obj["games"], 0)
                        print(f"[selfplay] {obj['games']}/{c.games_per_phase} games | "
                              f"{obj['evals_per_s']:.0f} evals/s | "
                              f"{obj['games_per_h']:.0f} g/h | ETA {eta/60:.0f}m")
                except Exception:
                    pass

        rc, out_lines = self.run_engine(args, on_line=on_line)
        if not self._stop and rc != 0:
            # surface the engine's own error lines (FATAL / terminate reasons)
            tail = [ln for ln in getattr(self, "_last_stderr_tail", [])][-6:]
            for ln in tail:
                print("[selfplay:engine]", ln)
            raise RuntimeError(f"selfplay exited {rc}")

        # Preferred stats source: the engine's authoritative summary JSON.
        summary = None
        for ln in reversed(out_lines):
            if '"type":"selfplay_summary"' in ln:
                try:
                    summary = json.loads(ln[ln.index("{"):])
                except Exception:
                    pass
                break

        if summary and summary.get("games"):
            g = summary["games"]
            total = max(summary["white_wins"] + summary["draws"]
                        + summary["black_wins"], 1)
            self.last_sp_stats = {
                "sp_games": g,
                "sp_positions": summary["positions"],
                "sp_avg_plies": round(summary["avg_plies"], 1),
                "sp_white_win": round(summary["white_wins"] / total, 3),
                "sp_draw": round(summary["draws"] / total, 3),
                "sp_black_win": round(summary["black_wins"] / total, 3),
                "sp_decisiveness": round(
                    (summary["white_wins"] + summary["black_wins"]) / total, 3),
                "sp_resigned": summary.get("resigned", 0),
            }
            for k, v in (summary.get("end_reasons") or {}).items():
                self.last_sp_stats[f"end_{k}"] = v
        else:
            # Fallback: scan the shards ourselves (also covers engine builds
            # older than the summary line). Warn loudly - zeros here are how
            # 12 iterations silently vanished last time.
            if rc == 0 and summary is None:
                print("[warn] no selfplay_summary from engine - falling back "
                      "to shard scan")
            pos = w = d = b = 0
            mat_sum = 0.0
            for p in set(out_dir.glob("*.gz")) - known_shards:
                n, ww, dd, bb, mm = scan_shard_results(p)
                pos += n; w += ww; d += dd; b += bb
                if n:
                    mat_sum += mm * n
            decided = w + b
            total = max(w + d + b, 1)
            self.last_sp_stats = {
                "sp_games": c.games_per_phase,
                "sp_positions": pos,
                "sp_avg_plies": round(pos / max(c.games_per_phase, 1), 1),
                "sp_white_win": round(w / total, 3),
                "sp_draw": round(d / total, 3),
                "sp_black_win": round(b / total, 3),
                "sp_decisiveness": round(decided / total, 3),
            }
        print("[selfplay] outcome stats:", json.dumps(self.last_sp_stats))
        d = self.last_sp_stats.get("sp_draw")
        if d is not None and d > 0.90:
            print(f"[selfplay] *** WARNING: draw rate {d:.1%} > 90% - the "
                  "draw spiral may be returning. Check: twofold_draw off, "
                  "max_plies high enough for adjudication, prior_plies > 0. "
                  "Run health_scan.py and watch top-1 visit share. ***")
        print(f"[selfplay] done in {(time.time()-t0)/60:.1f} min")

    def _bootstrap(self) -> Path:
        """First run: no nets at all -> export random-init net as champion."""
        if not self.champion.exists():
            self.export_candidate_if_needed()
            self.candidate.replace(self.champion)
        return self.champion

    def phase_train(self):
        """Returns 'ok' | 'insufficient' (caller decides next phase)."""
        # Same replay window the trainer will use - otherwise this outer check
        # passes while the trainer's windowed dataset comes back empty (the
        # insufficient_data spin-loop of the first run).
        ds = ShardDataset(str(self.data_dir),
                          max_cache_shards=self.cfg.train.cache_shards,
                          window_positions=self.cfg.train.replay_window)
        have = len(ds)
        if ds.refresh():
            have = len(ds)
        need = self.cfg.train.min_positions_to_start
        if have < need:
            print(f"[train] waiting for data: {have}/{need} positions")
            return "insufficient"

        from trainer.train import RichTrainer
        tr = RichTrainer(self.cfg, self.root)
        summary = tr.train_phases(phases=1)
        print("[train]", json.dumps(summary))
        return "ok" if summary.get("status") == "ok" else "insufficient"

    def _run_match(self, model_a: Path, model_b: Path) -> dict | None:
        """Play a gated match; returns the engine's match JSON or None."""
        c = self.cfg.gate

        def match_line(stream, line):
            if '"type":"matchprog"' in line:
                try:
                    obj = json.loads(line[line.index("{"):])
                    print(f"[gate] {obj['done']}/{obj['total']} games")
                except Exception:
                    pass

        rc, lines = self.run_engine([
            "match",
            "--model-a", str(model_a),
            "--model-b", str(model_b),
            "--games", str(c.games),
            "--visits", str(c.visits),
            "--batch", str(c.batch_size),
            "--threads", str(c.threads),
            "--games-per-worker", str(c.games_per_worker),
        ], on_line=match_line)
        if rc != 0:
            print(f"[gate] match exited {rc}: {model_a.name} vs {model_b.name}")
            return None
        for ln in reversed(lines):
            if '"type":"match"' in ln:
                try:
                    return json.loads(ln[ln.index("{"):])
                except Exception:
                    return None
        return None

    @staticmethod
    def _match_metrics(res: dict | None) -> dict:
        if not res:
            return {"score": None, "elo": None, "se": None}
        return {"score": res.get("score", 0.5), "elo": round(res.get("elo_diff", 0.0), 1),
                "se": round(res.get("elo_se", 0.0), 1)}

    def phase_gate(self):
        if not self.cfg.gate.enabled:
            print("[gate] disabled -> promoting candidate unconditionally")
            if self.candidate.exists():
                self.candidate.replace(self.champion)
            self.last_gate = {"gate_score": None, "gate_elo": None, "gate_se": None,
                              "anchor_score": None, "anchor_elo": None, "anchor_se": None,
                              "champion_anchor_elo": None}
            return

        c = self.cfg.gate
        anchor = self.root / c.anchor_model

        if not self.champion.exists():
            self.export_candidate_if_needed()
            self.candidate.replace(self.champion)
            if not anchor.exists():
                shutil.copyfile(self.champion, anchor)
                print(f"[gate] bootstrap: anchor frozen at {anchor}")
            self.state["champion_anchor"] = {"score": 0.5, "elo": 0.0}
            self.last_gate = {"gate_score": None, "gate_elo": None, "gate_se": None,
                              "anchor_score": 0.5, "anchor_elo": 0.0, "anchor_se": None,
                              "champion_anchor_elo": 0.0}
            return

        if not anchor.exists():
            # Legacy state without an anchor: freeze the current champion as the
            # reference point. From here on the anchor is never touched.
            shutil.copyfile(self.champion, anchor)
            self.state["champion_anchor"] = {"score": 0.5, "elo": 0.0}
            print(f"[gate] no anchor found -> froze current champion as {anchor}")

        if not self.candidate.exists():
            print("[gate] no candidate; skipping")
            self.last_gate = {"gate_score": None, "gate_elo": None, "gate_se": None,
                              "anchor_score": None, "anchor_elo": None, "anchor_se": None,
                              "champion_anchor_elo": None}
            return

        if c.measurement_only:
            # AlphaZero-faithful: single continually-updated network. The latest
            # export always becomes the champion (self-play uses the freshest
            # net next iteration); the anchor match is measurement only.
            print(f"[gate] measurement-only: candidate vs anchor ({anchor.name}), "
                  f"{c.games} games")
            t0 = time.time()
            res_a = self._match_metrics(self._run_match(self.candidate, anchor))
            print(f"[gate] match done in {(time.time()-t0)/60:.1f} min")
            self.state.setdefault("elo_history", []).append({
                "iter": self.state["iterations"],
                "score": None, "elo": None, "se": None,
                "anchor_score": res_a["score"], "anchor_elo": res_a["elo"],
            })
            self.candidate.replace(self.champion)
            self.state["consecutive_rejections"] = 0
            self.state["champion_anchor"] = {"score": res_a["score"], "elo": res_a["elo"]}
            self.last_gate = {"gate_score": None, "gate_elo": None, "gate_se": None,
                              "anchor_score": res_a["score"], "anchor_elo": res_a["elo"],
                              "anchor_se": res_a["se"],
                              "champion_anchor_elo": res_a["elo"]}
            print(f"[gate] PROMOTED unconditionally (paper mode) | "
                  f"anchor elo {res_a['elo']:+.1f}")
            return

        print(f"[gate] candidate vs champion: {c.games} games @ {c.visits} visits "
              f"({c.threads}x{c.games_per_worker} parallel)")
        t0 = time.time()
        res_c = self._match_metrics(self._run_match(self.candidate, self.champion))
        print(f"[gate] candidate vs anchor ({anchor.name}): {c.games} games")
        res_a = self._match_metrics(self._run_match(self.candidate, anchor))
        print(f"[gate] matches done in {(time.time()-t0)/60:.1f} min")

        champ_anchor = self.state.get("champion_anchor") or {"score": 0.0, "elo": 0.0}
        self.last_gate = {
            "gate_score": res_c["score"], "gate_elo": res_c["elo"], "gate_se": res_c["se"],
            "anchor_score": res_a["score"], "anchor_elo": res_a["elo"],
            "anchor_se": res_a["se"],
            "champion_anchor_elo": champ_anchor["elo"],
        }
        if res_c["score"] is None and res_a["score"] is None:
            print("[gate] both matches failed; skipping promotion this iteration")
            return

        self.state.setdefault("elo_history", []).append({
            "iter": self.state["iterations"],
            "score": res_c["score"], "elo": res_c["elo"], "se": res_c["se"],
            "anchor_score": res_a["score"], "anchor_elo": res_a["elo"],
        })

        score = res_c["score"]
        a_score = res_a["score"]
        if score is not None and score >= c.min_score:
            self.candidate.replace(self.champion)
            self.state["consecutive_rejections"] = 0
            self.state["champion_anchor"] = {"score": a_score, "elo": res_a["elo"]}
            print(f"[gate] PROMOTED (score {score:.3f} >= {c.min_score}) | "
                  f"anchor elo {res_a['elo']:+.1f}")
        else:
            rej = self.state.get("consecutive_rejections", 0) + 1
            self.state["consecutive_rejections"] = rej
            beats_champ_anchor = (a_score is not None
                                  and a_score >= champ_anchor["score"])
            if rej >= c.force_promote_after and beats_champ_anchor:
                # Anti-stall: the candidate keeps training from latest.pt every
                # iteration, so a merely noisy gate must not block forever - but
                # only force-promote a candidate that at least matches the
                # champion's own measured strength vs the SAME frozen anchor
                # (the old unconditional force-promote championed noise).
                self.candidate.replace(self.champion)
                self.state["consecutive_rejections"] = 0
                self.state["champion_anchor"] = {"score": a_score, "elo": res_a["elo"]}
                print(f"[gate] FORCE-PROMOTED after {rej} rejections "
                      f"(anchor {a_score:.3f} vs champion's {champ_anchor['score']:.3f})")
            else:
                why = "" if beats_champ_anchor or a_score is None else \
                    " (weaker than champion vs anchor - not force-promoting)"
                print(f"[gate] rejected (score {score} < {c.min_score}; "
                      f"{rej}/{c.force_promote_after} toward force-promotion){why}")

    # -- main loop -------------------------------------------------------------
    def run(self, max_iterations: int | None = None):
        consecutive_errors = 0
        while not self._stop:
            if max_iterations is not None and self.state["iterations"] >= max_iterations:
                break
            it = self.state["iterations"]
            print(f"\n=== iteration {it} ===")
            try:
                if self.state["phase"] == "selfplay":
                    self.phase_selfplay()
                    self.log_iter(getattr(self, "last_sp_stats", {}))
                    self.state["phase"] = "train"
                elif self.state["phase"] == "train":
                    outcome = self.phase_train()
                    if self._stop:
                        pass
                    elif outcome == "ok":
                        self.state["phase"] = "export"
                    else:
                        # not enough data (yet): go play more games instead of
                        # spinning forever on an empty dataset
                        print("[train] insufficient data -> back to selfplay")
                        self.state["phase"] = "selfplay"
                elif self.state["phase"] == "export":
                    self.export_candidate_if_needed()
                    self.state["phase"] = "gate"
                elif self.state["phase"] == "gate":
                    self.phase_gate()
                    self.log_iter(getattr(self, "last_gate",
                                          {"gate_score": None, "gate_elo": None,
                                           "gate_se": None}))
                    self.state["phase"] = "selfplay"
                    self.state["iterations"] = it + 1
                consecutive_errors = 0
            except Exception as e:   # noqa: BLE001
                consecutive_errors += 1
                print(f"[orchestrator] error ({consecutive_errors}): {e}")
                if consecutive_errors >= 5:
                    print("[orchestrator] 5 consecutive errors -> stopping "
                          "instead of spinning")
                    self._stop = True
                else:
                    time.sleep(5)
            self.save_state()
        self.save_state()
        print("[orchestrator] stopped cleanly")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--iterations", type=int, default=None,
                    help="stop after N full iterations (default: forever)")
    args = ap.parse_args()
    Orchestrator(Config.load(args.config)).run(max_iterations=args.iterations)
