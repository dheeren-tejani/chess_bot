# Roadmap

Where this project could go next, ranked by expected value per hour of
work. Everything here is a *hypothesis* — the project was stopped at
iteration 312 because it reached "working and deployed", not because
these experiments were exhausted.

---

## Near-term (days of work)

### 1. Retune hyperparameters at iter 300+

- **Status**: not started
- **Estimated gain**: 20–60 Elo
- **Cost**: 1 day of training + 2 match runs

The current config values were tuned for the iter-100 regime:

- `contempt: 0.40` — likely too low now that self-play is more accurate
- `wdl_draw_weight: 0.5` — probably fine but worth an A/B
- `resign_threshold: -0.85` — the value net is sharper now, so this
  probably fires on positions it shouldn't

**Method**: change one value at a time. Run one training iteration. Run
40 games vs SF-1700. Repeat. Keep whichever combination wins by >30 Elo
across three measurements.

**Why this is first**: cheapest possible improvement. No architecture
changes, no new data pipeline, no new code.

### 2. Deeper self-play (2048 visits)

- **Status**: not started
- **Estimated gain**: 30–80 Elo
- **Cost**: 2× training time per iteration

The WDL labels at 1024 visits have measurable noise. Halving the noise
means every training signal is more accurate, which should compound over
iterations.

**Method**:

```yaml
# configs/default.yaml
full_visits: 2048
fast_visits: 512
```

Then run one full iteration and compare the training loss curve against
the previous iteration's. If loss drops more steeply, the change is
working. If it doesn't, the bottleneck is capacity, not signal.

**Tradeoff**: iterations take ~2× longer. At 30–35 min per iteration,
this pushes to 60–70 min. Acceptable if you're willing to wait.

### 3. Verify the snapshot actually captures the engine subprocess

- **Status**: partially done
- **Estimated gain**: 15 s off cold start
- **Cost**: 2 hours of investigation

The deployment currently has `enable_memory_snapshot=True`, and the
Python interpreter state is clearly captured. But subprocess pipe FDs
may not survive restore — the 16 s first-request cold start on Modal
suggests the engine respawns.

**Method**: check whether `[modal] snapshot init complete` appears once
per container (good) or once per request (bad). If the latter, the
engine subprocess is being killed and respawned. The fix would be to
either (a) use a subprocess whose FDs survive Modal's snapshot restore,
or (b) accept the 16 s cost and keep `scaledown_window` at 600.

**Why not higher priority**: the cold start is a UX nuisance, not a
correctness issue, and 600s warm time covers most visitor patterns.

---

## Medium-term (weeks of work)

### 4. Bigger network (10 blocks × 160 channels)

- **Status**: not started
- **Estimated gain**: 80–200 Elo (cap depending on hyperparams)
- **Cost**: 3× training time per iteration, 2× inference time

The 6×96 net is at the small end of what's viable for competitive chess.
Leela's smallest useful net is ~10M parameters; this is 1.3M.

**Method**:

```yaml
# configs/default.yaml
net:
  blocks: 10
  channels: 160
```

Then re-export ONNX with matching `--blocks 10 --channels 160`. The
backend's batch size may need to drop from 16 to 8 to fit in memory,
though Modal's 16 GB should hold the larger model fine.

**Before doing this**: complete #1 and #2 first. Resizing the net without
first tuning hyperparameters risks discovering that the plateau was
signal-bound, not capacity-bound — in which case the bigger net does
nothing but cost 3× the compute.

### 5. Tree-parallel MCTS (shared tree + atomics)

- **Status**: not started
- **Estimated gain**: 2–4× search throughput
- **Cost**: 1–2 weeks of careful engineering

The current design uses root-parallel MCTS: K independent trees, each
single-threaded, sharing only a batcher. This is simple and correct, but
it wastes effort — every tree redundantly explores the root's children.

Tree-parallel MCTS uses a single shared tree with atomic visit counts
and per-edge locks. Every thread contributes to the same search,
eliminating the root-parallel tax and improving search quality per unit
of compute.

**Why this is risky**: it requires modifying `mcts.h` and `mcts.cpp`,
which are shared with self-play. The iter 237–269 regression happened
when these files were touched for an unrelated reason. Any tree-parallel
change needs the strict pre-merge test:

1. Build the engine.
2. Run 4-game self-play with `--seed 42` on old and new binaries.
3. Compare `end_reasons` counters. They should be identical if the
   change is truly behavior-preserving (which tree-parallel is not,
   so this test only confirms the change does something, not that it's
   correct).

Realistically: this is a project for a dedicated week, not an afternoon.

### 6. Opening book

- **Status**: not started
- **Estimated gain**: unclear, likely 0 for rating
- **Cost**: 1 day to build

The bot plays `d2d4` on move 1 every time. A shallow opening book (say,
top 5 lines by Lichess frequency) would add variety without changing
strength.

**Why it's low priority**: the bot is deployed and nobody's complained
about repetition. Also, "opening book from Lichess" implies external
data, which cuts against the tabula-rasa premise of the project.

---

## Long-term (research-y, unclear payoff)

### 7. Batch-1 UCI for evaluation parity

- **Status**: partially done
- **Estimated gain**: 30–80 Elo at fixed visit count (unclear at fixed time)
- **Cost**: 1 day

The current root-parallel UCI has a "root-parallel tax": at the same
visit count, K=32 shallow trees are slightly weaker than one deep tree.
The current config optimizes for latency (wall-clock) over visit-count
parity.

If you ever want to run matched games where both sides use the same
visit count (not the same time), you'd need a batch-1 UCI mode. This
would be slower but strictly comparable.

**Why it might not matter**: nobody runs matched games at fixed visit
count. Fixed time is what matters. The current design is correct for
the use case.

### 8. TensorRT inference

- **Status**: investigated, not implemented
- **Estimated gain**: 1.5–2× inference speedup → 5–10% end-to-end
- **Cost**: 1 day

TensorRT would speed up the ONNX inference step, but inference is only
~13% of per-move wall time on the 6×96 net. The dominant costs are tree
work, encoder, and batcher coordination, none of which TensorRT touches.

**Why it's low priority**: the gain doesn't justify the WSL2 CUDA
toolkit install, the engine rebuild with a new execution provider, or
the fp16-numerics risk. Revisit only if the network grows to 10×160 or
bigger, at which point inference becomes a larger fraction.

### 9. INT8 quantization

- **Status**: not started
- **Estimated gain**: potentially 2–3× inference, but often hurts policy
- **Cost**: 1–2 days

TensorRT supports INT8 with a calibration dataset. For chess policy
heads, INT8 typically degrades move ordering noticeably, which then
degrades search effectiveness. Worth testing only after all the
signal-side improvements (#1, #2) are exhausted.

---

## Explicitly out of scope

- **Distributed training**: doesn't fit the "single GPU, personal
  project" premise.
- **Support for a different game**: would require a different encoder,
  policy map, and search. Out of scope for this repository.
- **Public leaderboard or multiplayer**: the deployment is a
  single-visitor demo. Adding social features changes the project's
  scope from "engine" to "platform".
- **Continuous training in production**: the current design trains
  offline and deploys via ONNX. Online learning would require a
  completely different data path.

---

## How to prioritize

The order is:

- **#1** (retune) — cheapest, likely highest ROI
- **#2** (deeper self-play) — small config change, real gains expected
- **#4** (bigger net) — only after #1 and #2 have plateaued
- **#5** (tree-parallel) — only if search throughput becomes the
  bottleneck, which it probably won't for a portfolio project
- **Everything else** — as curiosity dictates

The honest framing: this project has already achieved its goal. The
roadmap exists because if you want to keep going, these are the
directions that make sense. If you don't want to keep going, none of
these are obligations.