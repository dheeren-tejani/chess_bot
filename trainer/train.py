"""Training loop: AMP(bf16), exact resume, wandb, rich UI, full metrics."""
from __future__ import annotations

import json
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.config import Config
from trainer.dataset import ShardDataset
from trainer.model import build_model


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


class BatchPrefetcher:
    """Async batch producer with bounded puts (never blocks forever on a full
    queue after stop()) and a thread-safe dataset-size snapshot."""
    def __init__(self, ds: ShardDataset, batch_size: int, seed: int, depth: int = 4,
                 refresh_every: int = 100):
        self.ds = ds
        self.bs = batch_size
        self.rng = np.random.default_rng(seed)
        self.refresh_every = refresh_every
        self.q = queue.Queue(maxsize=depth)
        self.stopped = threading.Event()
        self.last_len = len(ds)          # for metrics; avoids racing ds.refresh()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        step_count = 0
        while not self.stopped.is_set():
            step_count += 1
            if step_count % self.refresh_every == 0:
                try:
                    self.ds.refresh()
                except Exception:
                    pass
            batch = self.ds.sample_batch(self.bs, self.rng)
            if torch.cuda.is_available():
                batch = {k: v.pin_memory() if hasattr(v, "pin_memory") else v for k, v in batch.items()}
            # Retry put() with a timeout instead of blocking forever, so stop()
            # can always unblock us even if the consumer has already left.
            while not self.stopped.is_set():
                try:
                    self.q.put(batch, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def next(self) -> dict[str, torch.Tensor]:
        return self.q.get()

    def stop(self):
        self.stopped.set()
        # Drain anything sitting in (or about to land in) the queue so a
        # producer blocked in put() can observe the stop and exit.
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=5)


class RichTrainer:
    def __init__(self, cfg: Config, root: Path):
        self.cfg = cfg
        self.root = root
        self.ckpt_dir = root / cfg.run.checkpoint_dir
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.latest = self.ckpt_dir / "latest.pt"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_model = build_model({"blocks": cfg.net.blocks,
                                      "channels": cfg.net.channels}).to(self.device)
        self.opt = torch.optim.AdamW(self.raw_model.parameters(), lr=cfg.train.lr,
                                     weight_decay=cfg.train.weight_decay)
        self.base_lr = cfg.train.lr

        # Polyak EMA model weights tracking
        self.ema = {k: v.detach().clone().float() for k, v in self.raw_model.state_dict().items()}

        self.step = 0
        self.best_loss = float("inf")
        self.wandb_run_id = None
        self._load_resume()

        # Optional torch.compile for modern GPU setups
        self.model = self.raw_model
        if hasattr(torch, "compile") and self.device.type == "cuda":
            try:
                self.model = torch.compile(self.raw_model)
            except Exception as e:
                print(f"[warn] torch.compile skipped: {e}")

        self.wandb = None
        self.wandb_owner = False
        if cfg.wandb.enabled:
            try:
                import wandb
                if wandb.run is not None:
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
            except Exception as e:
                print(f"[warn] wandb unavailable ({e}); running offline")
                self.wandb = None
            self._setup_wandb_axes()

        self._rich = None
        try:
            from rich.console import Console
            from rich.table import Table
            self._rich = (Console(), Table)
        except Exception:
            pass

    def _load_resume(self):
        if not self.latest.exists():
            return
        try:
            ck = torch.load(self.latest, map_location=self.device, weights_only=False)
            self.raw_model.load_state_dict(ck["model"])
            self.opt.load_state_dict(ck["opt"])
            if "ema" in ck:
                self.ema = {k: v.to(self.device).float() for k, v in ck["ema"].items()}
            else:
                # checkpoint predates EMA tracking: seed EMA from the LOADED
                # weights, not the random init captured at construction
                self.ema = {k: v.detach().clone().float()
                            for k, v in self.raw_model.state_dict().items()}
            self.step = ck.get("step", 0)
            self.best_loss = ck.get("best_loss", float("inf"))
            self.wandb_run_id = ck.get("wandb_run_id")
            print(f"[resume] step {self.step} from {self.latest}")
        except Exception as e:
            print(f"[warn] failed to load {self.latest}: {e}; starting fresh")

    def save(self, rolling: bool = False):
        ck = {
            "model": self.raw_model.state_dict(),
            "ema": {k: v.cpu() for k, v in self.ema.items()},
            "opt": self.opt.state_dict(),
            "step": self.step,
            "best_loss": self.best_loss,
            "wandb_run_id": self.wandb_run_id,
            "net_cfg": {"blocks": self.cfg.net.blocks,
                        "channels": self.cfg.net.channels},
        }
        atomic_torch_save(ck, self.latest)
        if rolling:
            atomic_torch_save(ck, self.ckpt_dir / f"step_{self.step:08d}.pt")
            keep = max(int(self.cfg.train.keep_last_checkpoints), 1)
            olds = sorted(self.ckpt_dir.glob("step_*.pt"))
            for p in olds[:-keep]:
                p.unlink()

    TRAIN_KEYS = ("loss", "p_loss", "v_loss", "m_loss", "p_acc", "v_acc",
                  "v_acc_dec", "entropy", "gnorm", "lr", "pos/s",
                  "dataset_positions")

    def _setup_wandb_axes(self):
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
                    t.add_column(k, justify="right")
            row = [str(self.step)]
            for k in ("loss", "p_loss", "v_loss", "m_loss", "p_acc", "v_acc",
                      "v_acc_dec", "entropy", "gnorm", "lr", "pos/s"):
                if k in metrics:
                    v = metrics[k]
                    row.append(f"{v:.4g}" if isinstance(v, float) else str(v))
            t.add_row(*row)
            console.print(t)

    def lr_at(self, step):
        c = self.cfg.train
        local = step % c.steps_per_phase
        if local < c.warmup_steps:
            return c.lr * local / max(c.warmup_steps, 1)
        span = max(c.steps_per_phase - c.warmup_steps, 1)
        prog = (local - c.warmup_steps) / span
        return 0.1 * c.lr + 0.45 * c.lr * (1 + math.cos(math.pi * prog))

    def train_phases(self, phases: int = 1) -> dict:
        ds = ShardDataset(str(self.root / self.cfg.run.data_dir),
                          cache_bytes=self.cfg.train.cache_bytes,
                          window_positions=self.cfg.train.replay_window)
        ds.refresh()
        ds.prewarm(workers=8)  # Parallel decompress to fill cache once at startup
        if len(ds) < self.cfg.train.min_positions_to_start:
            return {"status": "insufficient_data",
                    "positions": len(ds),
                    "needed": self.cfg.train.min_positions_to_start}

        bs = self.cfg.train.batch_size
        prefetcher = BatchPrefetcher(ds, bs, seed=self.step + 12345, depth=4)
        start_step = self.step
        end_step = start_step + self.cfg.train.steps_per_phase * phases

        last_metrics = {}
        t0 = time.time()
        pos_seen = 0

        try:
            while self.step < end_step:
                lr = self.lr_at(self.step)
                for gparam in self.opt.param_groups:
                    gparam["lr"] = lr

                batch = prefetcher.next()
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
                    is_draw = wdl_t.argmax(dim=1) == 1
                    res_ce = torch.where(is_draw, res_ce * draw_w, res_ce)

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
                gnorm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), 4.0)
                self.opt.step()

                # Update Polyak EMA weights
                with torch.no_grad():
                    sd = self.raw_model.state_dict()
                    fkeys = [k for k in self.ema if self.ema[k].is_floating_point()]
                    ev = [self.ema[k] for k in fkeys]
                    rv = [sd[k] for k in fkeys]
                    torch._foreach_mul_(ev, 0.999)
                    torch._foreach_add_(ev, rv, alpha=0.001)
                    for k in self.ema:
                        if not self.ema[k].is_floating_point():
                            self.ema[k].copy_(sd[k])   # num_batches_tracked: exact

                self.step += 1
                pos_seen += bs

                with torch.no_grad():
                    best = logp.argmax(dim=1)
                    target_best = pol_t.argmax(dim=1)
                    p_acc = (best == target_best).float().mean().item()
                    v_pred = wdl.float().argmax(dim=1)
                    v_target = wdl_t.argmax(dim=1)
                    v_acc = (v_pred == v_target).float().mean().item()
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
                    "dataset_positions": prefetcher.last_len,
                }
                if v_acc_dec is not None:
                    last_metrics["v_acc_dec"] = v_acc_dec
                self.log(last_metrics)

                if self.step % self.cfg.train.checkpoint_every_steps == 0:
                    self.save(rolling=True)
        finally:
            prefetcher.stop()

        self.save()
        if self.wandb is not None and self.wandb_owner:
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