"""Training loop: AMP(bf16), exact resume, wandb, rich UI, full metrics."""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import Config
from trainer.dataset import ShardDataset
from trainer.model import build_model


# ---------------------------------------------------------------------------
def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class RichTrainer:
    def __init__(self, cfg: Config, root: Path):
        self.cfg = cfg
        self.root = root
        self.ckpt_dir = root / cfg.run.checkpoint_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.latest = self.ckpt_dir / "latest.pt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model({"blocks": cfg.net.blocks,
                                  "channels": cfg.net.channels}).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay)
        self.base_lr = cfg.train.lr

        self.step = 0
        self.best_loss = float("inf")
        self.wandb_run_id = None
        self.rng_state = None
        self._load_resume()

        # wandb (offline fallback keeps everything usable without login)
        self.wandb = None
        self.wandb_owner = False
        if cfg.wandb.enabled:
            try:
                import wandb
                if wandb.run is not None:
                    # Inside the orchestrator process: reuse ITS run instead of
                    # letting wandb.init() silently return it. All our curves
                    # go on a dedicated "trainer_step" x-axis (see _setup_axes)
                    # so they can never collide with the loop's iteration-axis
                    # metrics (this collision silently dropped most training
                    # curves of the first run).
                    self.wandb = wandb.run
                    self.wandb_owner = False
                else:
                    self.wandb = wandb.init(
                        project=cfg.wandb.project,
                        entity=cfg.wandb.entity,
                        name=f"{cfg.run.name}-net{cfg.net.blocks}x{cfg.net.channels}",
                        resume="allow",
                        id=self.wandb_run_id,
                        config=cfg.__dict__,
                    )
                    self.wandb_owner = True
                    if self.wandb_run_id is None and self.wandb and self.wandb.id:
                        self.wandb_run_id = self.wandb.id
            except Exception as e:   # noqa: BLE001
                print(f"[warn] wandb unavailable ({e}); running offline-free")
                os.environ.setdefault("WANDB_MODE", "offline")
                try:
                    import wandb
                    if wandb.run is None:
                        self.wandb = wandb.init(
                            project=cfg.wandb.project, resume="allow",
                            id=self.wandb_run_id, config=cfg.__dict__)
                        self.wandb_owner = True
                        if self.wandb_run_id is None and self.wandb and self.wandb.id:
                            self.wandb_run_id = self.wandb.id
                except Exception:
                    self.wandb = None
            self._setup_wandb_axes()

        self._rich = None
        try:
            from rich.console import Console
            from rich.table import Table
            self._rich = (Console(), Table)
        except Exception:
            pass

    # -- resume ------------------------------------------------------------
    def _load_resume(self):
        if not self.latest.exists():
            return
        try:
            ck = torch.load(self.latest, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            self.step = ck.get("step", 0)
            self.best_loss = ck.get("best_loss", float("inf"))
            self.wandb_run_id = ck.get("wandb_run_id")
            print(f"[resume] step {self.step} from {self.latest}")
        except Exception as e:   # noqa: BLE001
            print(f"[warn] failed to load {self.latest}: {e}; starting fresh")
            return
        # RNG state is best-effort only - never fatal
        try:
            rng = ck.get("rng") or {}
            if "torch" in rng:
                torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "python" in rng:
                random.setstate(rng["python"])
        except Exception as e:   # noqa: BLE001
            print(f"[warn] rng restore skipped ({e})")

    def save(self, rolling: bool = False):
        ck = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "best_loss": self.best_loss,
            "wandb_run_id": self.wandb_run_id,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "net_cfg": {"blocks": self.cfg.net.blocks,
                        "channels": self.cfg.net.channels},
        }
        atomic_torch_save(ck, self.latest)
        if rolling:
            # rolling snapshot so a corrupted latest.pt (or a bad training phase)
            # can be rolled back without losing the whole run
            atomic_torch_save(ck, self.ckpt_dir / f"step_{self.step:08d}.pt")
            keep = max(int(self.cfg.train.keep_last_checkpoints), 1)
            olds = sorted(self.ckpt_dir.glob("step_*.pt"))
            for p in olds[:-keep]:
                p.unlink()

    # -- logging -----------------------------------------------------------
    TRAIN_KEYS = ("loss", "p_loss", "v_loss", "m_loss", "p_acc", "v_acc",
                  "v_acc_dec", "entropy", "gnorm", "lr", "pos/s",
                  "dataset_positions")

    def _setup_wandb_axes(self):
        """Route training curves to a dedicated 'trainer_step' x-axis.

        wandb enforces monotonic global steps and silently DROPS out-of-order
        logs; explicit step= logging from resumed checkpoints (and sharing a
        run with the orchestrator) triggered exactly that. Custom axes have
        no monotonicity constraint, so resumed/overlapping history is safe.
        """
        if self.wandb is None:
            return
        try:
            import wandb
            wandb.define_metric("trainer_step", hidden=True)
            for k in self.TRAIN_KEYS:
                wandb.define_metric(k, step_metric="trainer_step")
        except Exception:
            pass

    def log(self, metrics: dict, force_table=False):
        if self.wandb is not None:
            try:
                self.wandb.log({"trainer_step": self.step, **metrics})
            except Exception:
                pass
        if self._rich is not None and (force_table or self.step % 50 == 0 or self.step <= 3):
            console, Table = self._rich
            t = Table(title=None, pad_edge=False, box=None)
            t.add_column("step", justify="right")
            for k in ("loss", "p_loss", "v_loss", "m_loss", "p_acc", "v_acc",
                      "v_acc_dec", "entropy", "gnorm", "lr", "pos/s"):
                if k in metrics:
                    v = metrics[k]
                    t.add_column(k, justify="right")
            row = [str(self.step)]
            for k in ("loss", "p_loss", "v_loss", "m_loss", "p_acc", "v_acc",
                      "v_acc_dec", "entropy", "gnorm", "lr", "pos/s"):
                if k in metrics:
                    v = metrics[k]
                    row.append(f"{v:.4g}" if isinstance(v, float) else str(v))
            t.add_row(*row)
            console.print(t)

    # -- lr schedule ---------------------------------------------------------
    def lr_at(self, step):
        """Warmup + cosine anneal, phase-aligned.

        Computed from the step's position WITHIN the current phase, so every
        phase starts at warmup and ends exactly at the 0.1*lr floor. (The old
        version anchored the cycle at global warmup_steps with span
        steps_per_phase - warmup, which drifted ~100 steps per phase and left
        most phases training at near-peak LR.)
        """
        c = self.cfg.train
        local = step % c.steps_per_phase
        if local < c.warmup_steps:
            return c.lr * local / max(c.warmup_steps, 1)
        span = max(c.steps_per_phase - c.warmup_steps, 1)
        prog = (local - c.warmup_steps) / span
        return 0.1 * c.lr + 0.45 * c.lr * (1 + math.cos(math.pi * prog))

    # -- main loop -----------------------------------------------------------
    def train_phases(self, phases: int = 1) -> dict:
        """Run `phases` training phases; returns summary of the last one."""
        ds = ShardDataset(str(self.root / self.cfg.run.data_dir),
                          max_cache_shards=self.cfg.train.cache_shards,
                          window_positions=self.cfg.train.replay_window)
        new_positions = ds.refresh()
        if len(ds) < self.cfg.train.min_positions_to_start:
            return {"status": "insufficient_data",
                    "positions": len(ds),
                    "needed": self.cfg.train.min_positions_to_start}

        rng = np.random.default_rng(seed=self.step + 12345)
        bs = self.cfg.train.batch_size
        start_step = self.step
        end_step = start_step + self.cfg.train.steps_per_phase * phases

        last_metrics = {}
        t0 = time.time()
        pos_seen = 0

        while self.step < end_step:
            if self.step % 200 == 0:
                got = ds.refresh()
                if got:
                    print(f"[data] +{got} positions (total {len(ds)})")

            lr = self.lr_at(self.step)
            for gparam in self.opt.param_groups:
                gparam["lr"] = lr

            batch = ds.sample_batch(bs, rng)
            planes = batch["planes"].to(self.device, non_blocking=True)
            scalars = batch["scalars"].to(self.device, non_blocking=True)
            pol_t = batch["policy_target"].to(self.device, non_blocking=True)
            wdl_t = batch["wdl_target"].to(self.device, non_blocking=True)
            mat_t = batch["material_target"].to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                policy, wdl, material = self.model(planes, scalars)

            logp = F.log_softmax(policy.float(), dim=1)
            p_loss = -(pol_t * logp).sum(dim=1).mean()

            logp_wdl = F.log_softmax(wdl.float(), dim=1)
            res_ce = -(wdl_t * logp_wdl).sum(dim=1)
            draw_w = self.cfg.train.wdl_draw_weight
            if draw_w != 1.0:
                # draw-result positions count less: with 80-99% draw rates the
                # WDL head otherwise converges to the constant "draw" predictor
                is_draw = wdl_t.argmax(dim=1) == 1
                res_ce = torch.where(is_draw, res_ce * draw_w, res_ce)

            # cold-start value shaping (TrainCfg.wdl_shape_beta): blend a
            # material prior into the WDL target, fading on an absolute-step
            # schedule. Teaches "material -> winning chances" from EVERY
            # position (dense) while honest outcome labels accumulate.
            # Deliberately NOT draw-weighted: its whole purpose is signal in
            # draw-dominated data. At beta=0 this reduces to res_ce exactly.
            beta = self.cfg.train.wdl_shape_beta
            if beta > 0 and self.cfg.train.wdl_shape_fade_end_step > 0:
                frac = ((self.cfg.train.wdl_shape_fade_end_step - self.step)
                        / max(self.cfg.train.wdl_shape_fade_span_steps, 1))
                beta *= max(0.0, min(1.0, frac))
            if beta > 0:
                m = (mat_t.clamp(-8, 8) / 8.0)
                shaped = torch.zeros_like(wdl_t)
                shaped[:, 0] = m.clamp(min=0)
                shaped[:, 2] = (-m).clamp(min=0)
                shaped[:, 1] = 1.0 - shaped[:, 0] - shaped[:, 2]
                shaped_ce = -(shaped * logp_wdl).sum(dim=1)
                per_pos_v = (1.0 - beta) * res_ce + beta * shaped_ce
            else:
                per_pos_v = res_ce
            v_loss = per_pos_v.mean()
            m_loss = F.mse_loss(material.float().squeeze(1),
                                mat_t.clamp(-15, 15) / 8.0)

            loss = (self.cfg.train.w_policy * p_loss +
                    self.cfg.train.w_wdl * v_loss +
                    self.cfg.train.w_material * m_loss)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
            self.opt.step()

            self.step += 1
            pos_seen += bs

            with torch.no_grad():
                best = logp.argmax(dim=1)
                target_best = pol_t.argmax(dim=1)
                p_acc = (best == target_best).float().mean().item()
                v_pred = wdl.float().argmax(dim=1)
                v_target = wdl_t.argmax(dim=1)
                v_acc = (v_pred == v_target).float().mean().item()
                # accuracy on DECISIVE positions only - overall v_acc is
                # inflated by the draw majority (99% there == "always draw")
                dec = v_target != 1
                v_acc_dec = ((v_pred == v_target)[dec]).float().mean().item() \
                    if bool(dec.any()) else None
                entropy = -(logp.exp() * logp).sum(dim=1).mean().item()

            last_metrics = {
                "loss": loss.item(),
                "p_loss": p_loss.item(),
                "v_loss": v_loss.item(),
                "m_loss": m_loss.item(),
                "p_acc": p_acc,
                "v_acc": v_acc,
                "entropy": entropy,
                "gnorm": float(gnorm),
                "lr": lr,
                "pos/s": pos_seen / max(time.time() - t0, 1e-9),
                "dataset_positions": len(ds),
            }
            if v_acc_dec is not None:
                last_metrics["v_acc_dec"] = v_acc_dec
            self.log(last_metrics)

            if self.step % self.cfg.train.checkpoint_every_steps == 0:
                self.save(rolling=True)

        self.save()
        if self.wandb is not None and self.wandb_owner:
            # never finish a run we borrowed from the orchestrator process
            try:
                self.wandb.finish()
            except Exception:
                pass
        return {"status": "ok", **last_metrics}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--phases", type=int, default=1)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    tr = RichTrainer(cfg, Path.cwd())
    out = tr.train_phases(phases=args.phases)
    print(json.dumps(out))
