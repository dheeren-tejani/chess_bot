"""Configuration loader (YAML -> nested dataclasses)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunCfg:
    name: str = "chessbot-v1"
    data_dir: str = "data/selfplay"
    checkpoint_dir: str = "checkpoints"
    model_dir: str = "models"


@dataclass
class NetCfg:
    blocks: int = 6
    channels: int = 96


@dataclass
class SelfPlayCfg:
    games_per_phase: int = 3000
    threads: int = 5
    games_per_worker: int = 3
    batch_size: int = 256
    fast_visits: int = 96
    full_visits: int = 512
    fast_prob: float = 0.75
    min_fresh_visits: int = 0          # Minimum newly searched visits required per move
    leaves_per_round: int = 16
    temperature_plies: int = 30
    prior_plies: int = 10              # sample first N plies from (noised) prior
    max_plies: int = 450
    shard_games: int = 24
    seed: int = 1
    use_twofold_draw: bool = True       # score search repetitions as draws
    # Ply-cap adjudication: |material diff| >= this (pawns) at max_plies scores
    # as a win for the leader instead of a draw. <= 0 disables (pure draw).
    adjudicate_material_pawns: float = 4.0
    # Cold-start aid: blend the material head into the self-play search value
    # (value += alpha * mat_pred/8). 0 disables. Reduce to 0 once the WDL head
    # discriminates on its own (see health probes: ev(queen-up) vs ev(start)).
    value_material_alpha: float = 0.0
    # Search-side draw value (training labels stay honest 0.5). Lower than 0.5
    # = contempt: both players avoid drifting into repetition/sterile draws.
    # Requires use_twofold_draw=true so repetitions are visible to the search.
    contempt: float = 0.5
    # Resignation (enable once the value net actually discriminates - enabling
    # it against a constant-value net just throws games away at random).
    resign_enabled: bool = False
    resign_threshold: float = -0.95
    resign_consecutive: int = 2
    resign_continue_frac: float = 0.05
    resign_min_ply: int = 30


@dataclass
class TrainCfg:
    steps_per_phase: int = 1200
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    min_positions_to_start: int = 25000
    replay_window: int = 3_000_000   # train only on the newest N positions
    w_policy: float = 1.0
    w_wdl: float = 1.0
    w_material: float = 0.1
    # Loss weight for positions whose game result was a DRAW. With 80-99% draw
    # rates the WDL head otherwise converges to the constant "draw" predictor
    # (the Aug-2026 collapse). < 1.0 keeps the rare decisive games dominant.
    wdl_draw_weight: float = 1.0
    # Cold-start value shaping: blend a material heuristic into the WDL target
    # (loss = (1-beta)*result_CE + beta*material_prior_CE). Teaches the value
    # head "material -> winning chances" from EVERY position while real
    # decisive labels accumulate. Fades linearly: full beta until
    # wdl_shape_fade_end_step - wdl_shape_fade_span_steps, then to 0 at
    # wdl_shape_fade_end_step. 0 disables.
    wdl_shape_beta: float = 0.0
    wdl_shape_fade_end_step: int = 0
    wdl_shape_fade_span_steps: int = 40000
    checkpoint_every_steps: int = 400
    keep_last_checkpoints: int = 3
    cache_shards: int = 256          # RAM-cached shards for the trainer dataset


@dataclass
class GateCfg:
    enabled: bool = True
    # AlphaZero-faithful mode (paper p.3: "AlphaZero simply maintains a single
    # neural network that is updated continually, rather than waiting for an
    # iteration to complete"). When true: no promotion decisions at all - the
    # candidate becomes the champion unconditionally each iteration (self-play
    # always uses the freshest net) and the anchor match runs purely as a
    # measurement of absolute progress. Eliminates promotion shocks and the
    # stale-champion stall.
    measurement_only: bool = False
    games: int = 24
    visits: int = 160
    batch_size: int = 128
    threads: int = 4
    games_per_worker: int = 3
    min_score: float = 0.55
    force_promote_after: int = 3   # consecutive rejections before forced promotion
    # Frozen random-init reference net. The candidate plays it every iteration;
    # anchor Elo is the ABSOLUTE progress curve (candidate-vs-champion alone can
    # hide regressions when champions churn). NEVER overwrite this file.
    anchor_model: str = "models/anchor.onnx"


@dataclass
class WandBCfg:
    enabled: bool = True
    project: str = "chess-bot"
    entity: str | None = None


@dataclass
class Config:
    run: RunCfg = field(default_factory=RunCfg)
    net: NetCfg = field(default_factory=NetCfg)
    selfplay: SelfPlayCfg = field(default_factory=SelfPlayCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    gate: GateCfg = field(default_factory=GateCfg)
    wandb: WandBCfg = field(default_factory=WandBCfg)

    @staticmethod
    def load(path: str | Path) -> "Config":
        cfg = Config()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for section in ("run", "net", "selfplay", "train", "gate", "wandb"):
            sub = raw.get(section) or {}
            target = getattr(cfg, section)
            for k, v in sub.items():
                if hasattr(target, k):
                    setattr(target, k, v)
        return cfg
