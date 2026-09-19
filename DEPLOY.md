# Deployment

This document covers the full deployment pipeline: from a trained `.pt`
checkpoint to a live browser-playable bot. It's written to be reproducible
from scratch on a fresh machine.

The deployment target is:

- **Backend**: Modal (CPU-only tier, free)
- **Frontend**: Netlify (free tier)
- **Model**: fp32 ONNX, embedded in the Modal image
- **Persistent storage**: Modal Volume for game records

Expected total cold-start time on a fresh machine: **~20 minutes**,
dominated by the first Modal image build.

---

## Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| Python | 3.10+ | Trainer + backend |
| CMake | 3.15+ | Engine build |
| C++ compiler | C++17 (g++ 9+ or clang 10+) | Engine build |
| Node.js | 18.14+ | Netlify CLI (optional) |
| Modal CLI | latest | `pip install modal` |
| `modal` account | free tier OK | Backend hosting |
| `netlify` account | free tier OK | Frontend hosting |

You also need a trained checkpoint. If you're starting from scratch, see
the main README for the training instructions.

---

## Part 1 — Export the ONNX models

Two variants are needed:

- **fp32** for CPU deployment (Modal)
- **fp16** for GPU deployment (if you ever move to a GPU region)

### fp32 (CPU)

```bash
cd ~/chess_bot
python trainer/export_onnx_cpu.py \
    --checkpoint checkpoints/latest.pt \
    --out models/champion.onnx \
    --verify
```

Expected output ends with `[verify] OK - exported model matches PyTorch.`
The resulting file is ~3 MB.

### fp16 (GPU, optional)

```bash
python trainer/export_onnx.py \
    --checkpoint checkpoints/latest.pt \
    --out models/champion_fp16.onnx \
    --blocks 6 --channels 96
```

### Why two files?

ONNX Runtime's CPU execution provider inserts fp16 -> fp32 -> fp16 cast
nodes at every layer boundary if the CPU lacks native fp16 arithmetic
(which Modal's fleet does). The fp32 export avoids ~2× slowdown from
these casts. The fp16 export is faster on GPU but not on CPU.

If you only deploy to one target, export only the corresponding file.

---

## Part 2 — Verify the engine locally

Before deploying, confirm the engine loads the exported model.

```bash
cd engine
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

engine/build/engine uci --model models/champion.onnx --visits 128 <<'EOF'
position startpos
go nodes 128
EOF
```

You should see a `bestmove` line and one `info` line. If you get
`model planes dtype mismatch`, the ONNX export produced the wrong input
type — re-export with the correct script.

### BMI2 portability check

The engine uses BMI2 (`_pext_u64`) for magic bitboard indexing. On
non-BMI2 CPUs it falls back to ray-walking attack generation — slower but
correct. Verify the guard is present:

```bash
grep -A3 "if (!BMI2)" engine/src/position.cpp
```

If that doesn't return anything, the fallback path is broken and the
engine will SIGILL on non-BMI2 CPUs. See commit history for the fix.

---

## Part 3 — Deploy the backend on Modal

### 3.1 — Set up secrets

Create `.env` at the repo root with the values the backend reads:

```bash
# .env — never commit this file
CHESS_VISITS=1024
CHESS_MOVETIME_MS=0
CHESS_DEVICE=cpu
CHESS_WORKERS=16
CHESS_BATCH=16
CHESS_ORT_THREADS=8
CHESS_CORS_ORIGINS=*
CHESS_DATA_DIR=/data
```

`modal.Secret.from_dotenv()` reads this at deploy time and injects every
key as an environment variable in the container.

**Do not** put `CHESS_MODEL_PATH` or `CHESS_ENGINE_BIN` in `.env` — those
are set explicitly in `modal_app.py`'s `setup()` to point at the
baked-in paths inside the image.

### 3.2 — Configure `modal_app.py`

The default config targets CPU-only free tier:

```python
@app.cls(
    image=image,
    cpu=16.0,                       # 16 physical cores
    memory=16384,                   # 16 GB
    enable_memory_snapshot=True,    # captured init state → fast cold start
    min_containers=0,               # scale to zero when idle
    max_containers=1,               # cap at one container (bounds cost)
    scaledown_window=600,           # keep warm 10 min after last request
    timeout=3600,
    volumes={"/data": games_volume},
    secrets=[chess_secrets],
    # Uncomment for India-adjacent routing. Costs ~1.15×.
    # region=["ap"],
    # routing_region="ap-south",
)
@modal.concurrent(max_inputs=10)
class ChessBot:
    ...
```

**Region caveat:** `routing_region` cannot be changed on an existing
function. If you want to add or remove it, you must `modal app stop chess-bot` and re-deploy.

### 3.3 — Deploy

```bash
modal deploy modal_app.py
```

First deploy takes 5–15 minutes (image build compiles the C++ engine,
installs Python deps, embeds the model). Subsequent deploys reuse cached
layers and take ~30 seconds.

Expected output ends with a URL like:

```text
✓ Created web endpoint: https://your-workspace--chess-bot-chessbot-web.modal.run
```

### 3.4 — Verify

```bash
URL="https://your-workspace--chess-bot-chessbot-web.modal.run"

# Health check
curl -s "$URL/healthz" | python3 -m json.tool

# Warm the container (first request pays cold start)
curl -s -o /dev/null -X POST "$URL/api/move" \
    -H 'Content-Type: application/json' \
    -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","moves":[]}'

# Measure warm latency
for i in 1 2 3; do
    curl -s -o /dev/null -w "req $i: %{time_total}s\n" \
        -X POST "$URL/api/move" \
        -H 'Content-Type: application/json' \
        -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","moves":[]}'
done
```

Check the logs for startup confirmation:

```bash
modal app logs chess-bot | grep -E "providers|intra-op|gpu_pref|visits"
```

Expected lines:

```text
[serve] ready | backend=cpp-uci visits=1024 movetime_ms=0 gpu_pref=False
[engine] [nn_client] ORT intra-op threads = 8 (gpu=0)
[engine] [nn_client] providers=[CPUExecutionProvider] gpu=NO batch=16
[modal] snapshot init complete; engine is warm
```

If `gpu_pref=True` appears when you asked for CPU, `CHESS_DEVICE` didn't
propagate. Check `.env` and `modal_app.py`'s `setup()` override.

### 3.5 — Set a spending cap

**Do this now.** Go to Modal dashboard → Settings → Billing → Spending
Limit. Set it to $5 or $10. This is the only defense that bounds your
worst-case cost if the endpoint is abused. `max_containers=1` limits
you to one container running, but the cap is what actually stops the
meter.

---

## Part 4 — Deploy the frontend on Netlify

### 4.1 — Build the frontend locally

```bash
cd frontend
npm install
npm run build
```

Confirm the build succeeds and the output lands in `dist/` or `build/`.

### 4.2 — Point the frontend at the proxy

If you're using a Netlify redirect (recommended — see §4.5), the frontend
should call a relative path, not the Modal URL directly:

```javascript
// src/api.ts (or wherever your API calls live)
const API_BASE = "/api/chess";
```

Then all requests go to `https://your-site.netlify.app/api/chess/...`,
which Netlify proxies to Modal. The Modal URL never appears in the
browser.

### 4.3 — Connect to Netlify

1. Log in to netlify.com
2. **Add new site** → **Import an existing project** → **GitHub**
3. Select your repository
4. Configure:
   - Base directory: `frontend/` (if the frontend lives in a subfolder)
   - Build command: `npm run build`
   - Publish directory: `frontend/dist` (or `frontend/build`)
5. Click **Deploy site**

### 4.4 — Set the proxy environment variable

In Netlify dashboard → Site settings → Environment variables, add:

| Key | Value |
|-----|-------|
| `MODAL_BACKEND_URL` | `https://your-workspace--chess-bot-chessbot-web.modal.run` |

This is used by the edge function in §4.5. It's never exposed to the
browser.

### 4.5 — Add the edge function proxy (recommended)

Create `netlify/edge-functions/chess-proxy.ts` in the frontend repo:

```typescript
export default async (request: Request) => {
  const MODAL_URL = Deno.env.get("MODAL_BACKEND_URL");
  if (!MODAL_URL) {
    return new Response(JSON.stringify({ error: "backend not configured" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }

  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/\.netlify\/functions\/chess-proxy/, "");
  const target = `${MODAL_URL}${path}${url.search}`;

  const body = ["GET", "HEAD"].includes(request.method)
    ? undefined
    : await request.text();

  const upstream = await fetch(target, {
    method: request.method,
    headers: {
      "content-type": request.headers.get("content-type") ?? "application/json",
    },
    body,
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
      "access-control-allow-origin": "*",
      "access-control-allow-methods": "GET, POST, OPTIONS",
      "access-control-allow-headers": "content-type",
    },
  });
};

export const config = { path: "/api/chess/*" };
```

Add the redirect in `netlify.toml` at the frontend repo root:

```toml
[[redirects]]
  from = "/api/chess/*"
  to = "/.netlify/functions/chess-proxy"
  status = 200

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

Commit and push. Netlify rebuilds and the proxy goes live.

**Why this matters:** without the proxy, anyone opening DevTools can see
your Modal URL and hit it directly, bypassing any frontend rate limiting.
With it, Modal is only reachable through Netlify.

### 4.6 — Verify

```bash
curl -X POST https://your-site.netlify.app/api/chess/api/move \
    -H 'Content-Type: application/json' \
    -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","moves":[]}'
```

If it returns a `best_move`, the full stack is live.

---

## Part 5 — Operational notes

### Cost

On Modal's CPU tier with `max_containers=1`, `min_containers=0`, and
`scaledown_window=600`:

| Traffic level | Approximate monthly cost |
|---------------|--------------------------|
| Zero visitors | $0 |
| ~20 sessions/day | ~$0.50 |
| Continuous use | ~$10–15 (16 cores × $0.047/hr) |

The spending cap you set in §3.5 is the hard ceiling.

### Cold starts

The first request after 10 minutes of idle pays a snapshot restore. With
`enable_memory_snapshot=True`, this is ~2 seconds for the Python
interpreter, but the C++ engine subprocess may respawn (~15 s) because
pipe file descriptors don't always survive snapshot restore.

If cold starts become a problem, `scaledown_window=1800` keeps the
container warm 30 minutes. Setting `min_containers=1` eliminates them
entirely but runs the container 24/7 — not worth the cost for a personal
project.

### Rate limiting

The backend has in-process rate limits in `backend/app.py`. Two buckets:

- 30 `/api/move` per IP per minute
- 10 `/api/games` saves per IP per minute

These are generous for human play (a person makes ~1 move per 5 seconds)
and restrictive for scrapers. The limits are per-container; if you ever
scale to multiple containers you'd need a shared store.

### Volume persistence

Game records are written to `/data/games/*.json` inside the Modal
Volume. Modal auto-commits volumes every few seconds and on container
shutdown, so no explicit `commit()` call is needed for low-frequency
writes. If you ever observe lost saves, add an explicit commit in
`_atomic_write_json`.

### Updating the model

To deploy a new checkpoint:

```bash
# Export the new ONNX
python trainer/export_onnx_cpu.py \
    --checkpoint checkpoints/latest.pt \
    --out models/champion.onnx

# Redeploy (Modal detects the changed file and rebuilds the layer)
touch models/champion.onnx
modal deploy modal_app.py
```

The `touch` ensures the file's mtime changes, which forces Modal to
rebuild the image layer that copies it in. Without the `touch`, a
redeploy with unchanged file contents would reuse the cached layer and
serve the old model.

---

## Troubleshooting

### `model planes dtype mismatch` at engine startup

The ONNX you're loading has `uint8` input, but `nn_client.cpp` expects
`int32`. Either:

1. Re-export with `export_onnx_cpu.py` (produces `uint8`, matching the
   original engine), or
2. Revert `nn_client.cpp` to the `uint8` path if you had previously
   changed it for TensorRT.

### SIGILL at engine startup

BMI2 not supported on the host CPU, and the guard in `attacks::init()` is
missing or broken. See the "BMI2 portability check" in Part 2.

### `engine EOF waiting for 'uciok'`

The engine subprocess died before responding to the initial handshake.
Most common causes:

- Model path wrong (check `modal_app.py` `setup` — should be
  `/root/models/champion.onnx`)
- Missing shared library (`libonnxruntime.so`) — verify with
  `modal app logs chess-bot | grep -i "cannot open shared"`
- SIGILL from missing BMI2 guard (see above)

### First request takes 15+ seconds

That's the cold start. Expected on the first visit after 10 minutes of
idle. If it's happening on every request, the snapshot isn't working —
check that `enable_memory_snapshot=True` is on the `@app.cls` and that
`setup()` is decorated with `@modal.enter(snap=True)`.

### Requests succeed but move quality looks wrong

Check the engine's reported eval. On a balanced opening position it
should be within ±30 cp. If it reports ±1040 (the clip ceiling)
consistently, the model isn't being loaded correctly — verify the ONNX
path and the model file's hash.

---

## Reference: full command sequence

For a fresh deploy from a working training run:

```bash
# 1. Export models
python trainer/export_onnx_cpu.py --checkpoint checkpoints/latest.pt \
    --out models/champion.onnx --verify

# 2. Verify engine loads it
engine/build/engine uci --model models/champion.onnx --visits 128 <<'EOF'
position startpos
go nodes 128
EOF

# 3. Deploy backend
modal deploy modal_app.py

# 4. Smoke test backend
URL="https://your-workspace--chess-bot-chessbot-web.modal.run"
curl -s "$URL/healthz" | python3 -m json.tool

# 5. Deploy frontend (assumes Git-connected Netlify site)
cd frontend
git add -A
git commit -m "Deploy: point at production backend"
git push

# 6. Verify end-to-end
curl -X POST https://your-site.netlify.app/api/chess/api/move \
    -H 'Content-Type: application/json' \
    -d '{"fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","moves":[]}'
```