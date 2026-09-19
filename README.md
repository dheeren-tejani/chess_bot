# Tabula Rasa Chess

> A self-play chess engine written from scratch — C++ search, PyTorch
> training, ONNX inference, deployed as a browser-playable bot.

[![Live demo](https://img.shields.io/badge/demo-play%20now-blue)](YOUR_LIVE_URL)
[![Engine](https://img.shields.io/badge/engine-C%2B%2B-brightgreen)](#architecture)
[![Model](https://img.shields.io/badge/model-PyTorch-orange)](#training)
[![Deploy](https://img.shields.io/badge/deploy-Modal%20CPU-purple)](#deployment)

**Play it:** [Chess Bot](https://chessbotplay.netlify.app/)

---

## What this is

An AlphaZero-style chess engine trained entirely from self-play, with no
human games, no opening books, and no handcrafted evaluation. Every
component — the bitboard engine, the MCTS search, the neural network, the
training loop, the data pipeline, the inference server — was written from
scratch as a personal project.

The final model plays at roughly **1600 FIDE** (~SF-1700 with 1000 ms per
move), converted from a 312-iteration self-play run on a single consumer
GPU. It was stopped by choice when the project reached "working, fast, and
deployable", not because the architecture hit a ceiling.

## Strength

Measured against Stockfish's calibrated `UCI_Elo` ladder at 1000 ms/move
per side:

| Iteration | Opponent | Score | Elo diff |
|-----------|----------|-------|----------|
| 44        | SF-0     | 21.5% | −225     |
| 84        | SF-0     | 35.5% | −104     |
| 126       | SF-0     | 43.5% | −45      |
| 176       | SF-0     | 55.0% | +35      |
| 234       | SF-1500  | 50.0% | parity   |
| 312       | SF-1700  | 32.5% | −127 ± 37 |

Approximate human-rating equivalents for the final checkpoint:

| Scale             | Estimated rating |
|-------------------|------------------|
| FIDE              | ~1600            |
| Lichess Classical | ~1880            |
| Chess.com Rapid   | ~1520            |

These conversions are approximate — the `UCI_Elo` ladder, FIDE, and
online rating pools are not calibrated to each other. Treat them as
buckets, not decimals.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                            ENGINE (C++17)                            │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │   Position   │ →  │  MCTS Search │ →  │  NN Client   │            │
│  │  (bitboards) │    │(root-parallel│    │  (ONNX RT)   │            │
│  └──────────────┘    │  + batcher)  │    └──────────────┘            │
│                      └──────────────┘                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │  Self-play   │ →  │  ShardWriter │ →  │  UCI driver  │            │
│  │ (12 threads) │    │    (gzip)    │    │ (HTTP-ready) │            │
│  └──────────────┘    └──────────────┘    └──────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           TRAINER (Python)                           │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │ ShardDataset │ →  │   ChessNet   │ →  │  Checkpoints │            │
│  │  (lazy load) │    │ 6×96 SE-Res  │    │  (.pt, EMA)  │            │
│  └──────────────┘    └──────────────┘    └──────────────┘            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │   Trainer    │    │  ONNX Export │    │ Orchestrator │            │
│  │  (AMP + EMA) │    │  (fp32/fp16) │    │ (self→train) │            │
│  └──────────────┘    └──────────────┘    └──────────────┘            │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       SERVE (FastAPI + Modal)                        │
│        Persistent C++ subprocess → HTTP API → Browser UI             │
└──────────────────────────────────────────────────────────────────────┘
```

## The pipeline

The orchestrator runs an endless loop:

```text
self-play (1200 games) → train (2400 steps) → export ONNX
      ▲                                            │
      │                                            ▼
      └───────── promote candidate ← gate/measure ─┘
```

Each phase runs as a separate subprocess so a crash in one doesn't corrupt
the others. Progress lives in `orchestrator_state.json` and the loop is
fully resumable.

### Self-play

The C++ engine plays 12 games in parallel across 12 threads. Each game uses
one of two visit budgets (256 or 1024, mixed probabilistically), with root
Dirichlet noise and move sampling for the first 30 plies, then greedy
play. Positions are written as gzipped binary shards — 128 games per shard,
~950 KB each.

Key search features:

- **Twofold draw pruning**: a search-internal single repetition is scored
  as a draw to prevent the self-play drift into threefold repetition that
  afflicts naive AlphaZero implementations.
- **Contempt**: the search-side draw value is 0.40 instead of 0.50, so
  both sides avoid sterile lines. Training labels remain honest (0.5).
- **Material shaping** (cold-start only): a fraction of the auxiliary
  material head is blended into the search value for the first ~65k steps
  to give the WDL head a dense learning signal before decisive games
  accumulate.

### Training

`train.py` implements:

- AdamW + cosine LR with warm restarts (SGDR, 2400 steps/phase)
- bf16 autocast on CUDA
- Polyak EMA (decay 0.999) tracked alongside the raw model
- WDL loss with down-weighted draws (`wdl_draw_weight=0.5`)
- Cold-start material-into-WDL shaping that fades linearly to zero

Shards are loaded lazily with a byte-budgeted LRU cache and a replay
window (1.5M positions = ~10 iterations of fresh data).

### Export

`export_onnx.py` produces an fp16 ONNX with a `BitPackedInput` wrapper
that accepts `uint8` planes and unpacks them via a LUT inside the graph.
`export_onnx_cpu.py` produces the fp32 variant used for CPU deployment.
The 3 MB fp32 model is embedded directly into the deployment image.

## Technical highlights

### Position encoding (113 planes + 9 scalars)

Spatial input is 8 ages × 14 planes each (112), plus an en-passant plane:

```text
age 0 (current position):
  planes 0..11   piece bitboards (white P,N,B,R,Q,K then black)
  plane  12      position occurred ≥1 extra time in history
  plane  13      position occurred ≥2 extra times
age 1..7 (previous positions, oldest missing = zero planes)
  same 14-plane layout
plane 112        en-passant target square (one-hot)
```

Scalars capture halfmove clock, fullmove number, repetition count, side to
move, four castling rights bits, and an EP-exists flag.

### Policy head (73 × 64 = 4672 logits)

AlphaZero-style layout, relative to the mover:

- **Planes 0–55**: queen-like moves, 8 directions × 7 distances
- **Planes 56–63**: knight moves, 8 deltas in a fixed order
- **Planes 64–72**: underpromotions (3 pieces × 3 capture directions)

Promotion to queen rides the normal queen planes. The vertical axis is
mirrored for black, so a white queen move up-left and a black queen move
down-right share the same plane.

### Root-parallel MCTS

The biggest engineering effort in the project was making the search fast
enough for interactive play. The final design runs **K independent
searches of the same root** in K threads, all sharing one batcher. Each
worker produces one leaf per round; the batcher aggregates all K into a
single ONNX batch.

This is architecturally different from vanilla MCTS and comes with a
tradeoff: at a fixed visit count, K parallel shallow trees are slightly
weaker than one deep tree (the "root-parallel tax", roughly 30–80 Elo).
But at a fixed *wall-clock* budget, K-way parallelism wins by a wide
margin because it can actually use the GPU.

Empirically, K=32 with a matching batch size is ~8–10× faster than the
naive single-threaded search at the same visit count. A 1024-visit move
dropped from ~2080 ms to ~250 ms on an RTX 3050 Laptop.

### Compressed shard format

Training data is written as gzipped binary records with a 20-byte header:

```text
magic "CBSP" | version u32 | games u32 | model_hash u64
```

Each record is a fixed-size struct:

```text
planes  : uint64[113]                          (896 B)
scalars : float32[9]                           (36 B)
policy  : {u16 idx, u16 visits} × 64           (256 B)
meta    : u8 result, u16 material_f16, u8 stm  (4 B)
```

Fixed-size records mean random access is O(1) — the dataset peeks at the
gzip ISIZE footer to compute record counts without decompressing.

## Deployment

The bot runs on **Modal's free CPU tier**. The C++ engine is compiled
inside the image at deploy time, the fp32 ONNX is embedded, and game
records are persisted to a Modal Volume.

Key deployment details:

- **Memory Snapshots**: initialization (ONNX load, engine warmup) is
  captured once and restored on every cold start, cutting cold-start
  latency from ~15 s to ~2 s.
- **Persistent container**: `allow_concurrent_inputs=True` keeps the ASGI
  app alive between requests, so the C++ subprocess isn't respawned per
  move.
- **Region routing**: `routing_region="ap-south"` routes Indian traffic
  through Modal's Mumbai edge.
- **CPU-optimized model**: the fp32 ONNX variant avoids the fp16
  cast-node penalty that ONNX Runtime's CPU execution provider incurs on
  hardware without native fp16 support.

Full deploy instructions are in [DEPLOY.md](DEPLOY.md).

### Performance (final deployment)

| Metric                                | Value       |
|---------------------------------------|-------------|
| Engine time per move (1024 visits)    | ~600 ms     |
| Client round-trip (India → Modal)     | ~1400 ms    |
| Cold start (first visit after idle)   | ~2 s        |
| Concurrent users supported            | 1 (serialized by engine lock) |

## Project structure

```text
chess_bot/
├── engine/                 C++17 chess engine + MCTS + UCI
│   ├── src/
│   │   ├── types.h         Move, Color, PieceType, packed encoding
│   │   ├── position.h/.cpp Bitboard position, move gen, perft
│   │   ├── zobrist.h       Deterministic Zobrist keys
│   │   ├── encoder.h/.cpp  113-plane position encoding
│   │   ├── policy_map.h/.cpp Move ↔ policy index
│   │   ├── mcts.h/.cpp     Tree, Search, PUCT, root-parallel
│   │   ├── nn_client.h/.cpp ONNX Runtime wrapper
│   │   ├── selfplay.h/.cpp Batcher, ShardWriter, game loop
│   │   ├── uci.h/.cpp      Interactive UCI driver
│   │   └── main.cpp        Subcommand dispatcher
│   └── tests/              Perft verification against python-chess
├── trainer/                Python training pipeline
│   ├── model.py            ChessNet (SE-ResBlocks, 3 heads)
│   ├── dataset.py          Shard loader with LRU cache
│   ├── train.py            Training loop (AMP, EMA, SGDR)
│   ├── export_onnx.py      GPU fp16 export
│   ├── export_onnx_cpu.py  CPU fp32 export
│   ├── encoding.py         Python mirror of C++ encoding
│   ├── config.py           YAML config loader
│   └── orchestrator.py     self-play → train → export loop
├── backend/                FastAPI server
│   └── app.py              C++ UCI subprocess + HTTP API
├── configs/
│   ├── default.yaml        Full training config
│   └── smoke.yaml          Tiny config for quick validation
├── modal_app.py            Modal deployment (CPU, snapshots, volume)
└── models/
    ├── champion.onnx       Final fp32 model (3 MB)
    └── champion_fp16.onnx  GPU variant
```

## Running locally

### Build the engine

```bash
cd engine
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The engine expects ONNX Runtime at `../../third_party/onnxruntime/`. Get
it from the official releases:

```bash
cd third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
tar -xzf onnxruntime-linux-x64-1.20.1.tgz
mv onnxruntime-linux-x64-1.20.1 onnxruntime
```

### Verify the engine

```bash
# Standard perft suite
./engine perft "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1" 1 2 3 4

# Compare against python-chess
python tests/bisect_pos6.py "r4rk1/1pp1qppp/p1np1n2/2b1p1b1/2B1P1B1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10" 3
```

### Play against the engine (UCI)

```bash
./engine uci --model models/champion.onnx --visits 1024
```

### Run a self-play iteration

```bash
./engine selfplay \
    --model models/champion.onnx \
    --out data/selfplay/iter_0000 \
    --games 100 --threads 8 --batch 256 \
    --fast-visits 256 --full-visits 1024
```

### Train

```bash
python trainer/train.py --config configs/default.yaml --phases 1
```

### Run the full loop

```bash
python trainer/orchestrator.py --config configs/default.yaml
```

### Serve

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Then curl the API:

```bash
curl -X POST http://localhost:8000/api/move \
    -H 'Content-Type: application/json' \
    -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","moves":[]}'
```

## Engineering lessons

A few things this project taught me that aren't obvious until you hit
them:

1. **Root-parallel MCTS is not a free lunch.** At fixed visit count, K
   parallel trees search wider but shallower than one deep tree. The
   tradeoff is worth it for interactive play (you get more than enough
   extra visits/second to compensate), but it means a K=32 engine at 1000
   visits is measurably weaker than a single-threaded engine at 1000 visits.
   Root-parallel is a latency optimization, not a strength optimization.

2. **fp16 is a GPU-only optimization.** A model exported as fp16 runs fast
   on CUDA but catastrophically slow on ONNX Runtime's CPU execution
   provider, because the runtime inserts fp16→fp32→fp16 cast nodes at every
   layer boundary when the CPU lacks native fp16 arithmetic. Serving the
   same fp16 model on Modal's CPU fleet was ~2× slower than the fp32 version.
   Always export a CPU variant if the deploy target is CPU.

3. **ONNX Runtime's `intra_op_num_threads` is a footgun.** The default is
   "all available cores", which is wrong for a workload like ours where many
   threads are already running independent searches. Setting it to 1 for GPU
   mode and 8 for CPU mode gave a 2.4× speedup on the deployed CPU version.

4. **BMI2/PEXT is not universal.** The bitboard engine uses `_pext_u64` to
   index attack tables, which is an instruction from Intel's BMI2 extension.
   AMD and older Intel chips before Haswell don't have it. The original code
   called it unconditionally during init, which crashed the engine with a
   SIGILL on Modal's Mumbai region. Guarding the table-build behind a CPU
   feature check fixed it — the fallback path uses slow ray-walking
   attack generation.

5. **A 33-iteration regression is a 33-iteration cost.** At iteration 237,
   I modified `mcts.cpp` and `position.cpp` while working on an unrelated
   performance change and introduced a subtle bug that caused self-play to
   drift into a draw spiral. I didn't notice for 33 iterations. Reverting
   cost hours of training. The lesson: any change to a file that self-play
   uses needs a smoke test (a 4-game run with identical seeds, comparing
   `end_reasons` counters) before the training loop touches it. `uci.cpp`
   touches nothing downstream — safe to iterate on. `mcts.cpp` and
   `position.cpp` are shared — risky.

## What I'd do next

If I continued this project, the ranking of next steps would be:

1. **Bigger net** (10 blocks × 160 channels instead of 6 × 96). The
   architecture has headroom; the current ceiling is somewhere around
   1700–1800 FIDE and a larger net would push that.

2. **Deeper self-play** (2048 full visits instead of 1024). The WDL
   labels at 1024 visits have measurable noise; halving it would double
   the accuracy of each training signal.

3. **Hyperparameter retune at iter 300+.** `contempt`, `resign_threshold`,
   and `wdl_draw_weight` were tuned for the iter-100 range and are likely
   suboptimal now.

4. **Tree-parallel MCTS** (shared tree + spinlocks). Another 2–4× search
   speedup, but architecturally invasive.

Full roadmap in [ROADMAP.md](ROADMAP.md).

## Acknowledgements

- **AlphaZero** (Silver et al., 2017) for the training paradigm.
- **Leela Chess Zero** for demonstrating the paradigm works at scale in
  chess, and for many of the self-play stabilization tricks (cpuct,
  FPU, twofold pruning) that are now standard.
- **python-chess** for the reference implementation used in perft tests.
- **Stockfish** for the `UCI_Elo` ladder, which gave this project an
  external yardstick.

## License

MIT. See [LICENSE](LICENSE).

---

*Built from scratch over several weeks as a personal project. The full
training trajectory (312 iterations, ~470k self-play games, ~52M
positions) ran on a single RTX 3050 Laptop GPU (4GB VRAM).*