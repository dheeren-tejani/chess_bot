# Tabula-Rasa Chess Bot

AlphaZero-style chess engine trained purely via self-play on an RTX 3050 Laptop GPU.
No human games, no openings book - the net develops its own play style.

## Architecture

```
engine/    C++17  bitboard movegen (PEXT), batched MCTS (virtual loss, FPU,
           playout-cap randomization), ONNX Runtime CUDA inference,
           gzip shard writer
trainer/   Python PyTorch ResNet+SE policy/WDL/material net, bf16 AMP training,
           fp16 ONNX export, exact-resume checkpoints, wandb + rich logging
tests/     perft suite, C++<->Python encoder/policy parity, dataset round-trip
```

## Setup (once)

```bash
# toolchain + python deps already installed; ORT C++ lib in third_party/
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j 12
```

## Running

```bash
# one full loop iteration: selfplay -> train -> export -> gate
./chess/bin/python trainer/orchestrator.py --config configs/default.yaml --iterations 1

# forever (Ctrl-C saves state; rerun resumes exactly where it stopped)
./chess/bin/python trainer/orchestrator.py --config configs/default.yaml
```

Components individually:

```bash
# rules validation (must match published node counts)
engine/build/engine perft "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 5
bash engine/tests/run_perft.sh

# GPU throughput of a net
scripts/engine_env.sh engine/build/engine bench --model models/champion.onnx

# manual self-play generation
scripts/engine_env.sh engine/build/engine selfplay --model models/champion.onnx \
    --out data/selfplay --games 1000 --threads 5

# head-to-head with Elo estimate
scripts/engine_env.sh engine/build/engine match --model-a A.onnx --model-b B.onnx --games 40

# training only
./chess/bin/python trainer/train.py --config configs/default.yaml --phases 1

# test suite
./chess/bin/python -m pytest tests/ -q
```

## Key design points

- **Copy-make positions** (~128 B) - no undo stacks; MCTS threads never share trees.
- **Single-writer-per-tree**: the GPU feeder only fills eval tasks; each worker
  applies its own completions through per-worker mailboxes. No locks in search.
- **Playout cap randomization** (KataGo): 75% of moves get `fast_visits`, the rest
  full budget - most of the speed win, negligible strength cost.
- **WDL value head** + auxiliary material head (KataGo-style extra gradient signal).
- **Twofold repetitions inside search score as draws**, teaching the net to shun
  repetitions without waiting for threefold.
- **Shard format**: fixed 1172 B records, gzip, top-visits-64 policy storage,
  stm byte for WDL target orientation. Trainer samples uniformly over the whole
  window and tracks consumed positions in a ledger for exact resume.

## Scaling up later

The pipeline is hardware-agnostic: rent a bigger GPU, point the same commands at
it, bump `net.blocks/channels` in the config, resume from checkpoints. Nets are
progressively growable (6x96 -> 10x160 -> ...) by re-training from any checkpoint.
