import sys
from pathlib import Path
import numpy as np, onnx, torch
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trainer.model import build_model

ONNX_IN, OUT = "models/safe_models/champion_iter_236.onnx", "checkpoints/latest.pt"
BLOCKS, CHANNELS, STEP, BN_EPS = 6, 96, 566_400, 1e-5

model = build_model({"blocks": BLOCKS, "channels": CHANNELS})
sd = model.state_dict()

m = onnx.load(ONNX_IN)
inits_raw = {i.name: numpy_helper.to_array(i).astype(np.float32)
             for i in m.graph.initializer}          # dict preserves file order

renamed = [a for n, a in inits_raw.items() if n.startswith("onnx::")]
named = {n[4:] if n.startswith("net.") else n: a
         for n, a in inits_raw.items() if not n.startswith("onnx::")}
assert len(named) == 34, sorted(named)[:5]

# Renamed tensors alternate (folded_weight, folded_bias) in execution order.
pairs = []
i = 0
while i < len(renamed):
    w, b = renamed[i], renamed[i + 1]
    assert w.ndim == 4 and b.ndim == 1 and b.shape[0] == w.shape[0], (i, w.shape, b.shape)
    pairs.append((w, b)); i += 2
assert len(pairs) == 16, len(pairs)

pair_specs = [("stem.0", "stem.1")]
for i in range(BLOCKS):
    pair_specs += [(f"body.{i}.c1", f"body.{i}.b1"), (f"body.{i}.c2", f"body.{i}.b2")]
pair_specs += [("p_head.0", "p_head.1"), ("v_head.0", "v_head.1"), ("m_head.0", "m_head.1")]

def build(pair_list):
    new = {}
    for (w, b), (conv, bn) in zip(pair_list, pair_specs):
        C = w.shape[0]
        new[conv + ".weight"] = torch.from_numpy(w.copy())
        new[bn + ".weight"] = torch.ones(C)
        new[bn + ".bias"] = torch.from_numpy(b.copy())
        new[bn + ".running_mean"] = torch.zeros(C)
        new[bn + ".running_var"] = torch.full((C,), 1.0 - BN_EPS)
    for k, v in sd.items():
        if v.is_floating_point() and k not in new:
            assert k in named, f"unmapped: {k}"
            assert tuple(named[k].shape) == tuple(v.shape), (k, named[k].shape, v.shape)
            new[k] = torch.from_numpy(named[k].copy())
        elif not v.is_floating_point():
            new[k] = torch.zeros_like(v)          # num_batches_tracked
    return new

# --- verify: rebuilt model (eval) vs the onnx itself ---
import onnxruntime as ort
try:
    sess = ort.InferenceSession(ONNX_IN, providers=["CPUExecutionProvider"])
except Exception:
    sess = ort.InferenceSession(ONNX_IN, providers=["CUDAExecutionProvider"])
rng = np.random.default_rng(0)

def compare(md):
    md.eval()
    maxp = maxw = 0.0; agree = 0
    with torch.no_grad():
        for _ in range(8):
            planes = rng.integers(0, 256, (32, 113, 8), dtype=np.uint8)
            sc = rng.random((32, 9)).astype(np.float32)
            po, wo, _ = sess.run(None, {"planes": planes, "scalars": sc})
            bits = np.unpackbits(planes, axis=2, bitorder="little").reshape(32, 113, 8, 8)
            pol, wdl, _ = md(torch.from_numpy(bits).float(), torch.from_numpy(sc))
            pp = torch.softmax(pol, 1).numpy(); ww = torch.softmax(wdl, 1).numpy()
            op = np.exp(po - po.max(1, keepdims=True)); op /= op.sum(1, keepdims=True)
            ow = np.exp(wo - wo.max(1, keepdims=True)); ow /= ow.sum(1, keepdims=True)
            maxp = max(maxp, np.abs(pp - op).max())
            maxw = max(maxw, np.abs(ww - ow).max())
            agree += int((pol.argmax(1).numpy() == po.argmax(1)).sum())
    return maxp, maxw, agree / 256

new_sd = build(pairs)
model.load_state_dict(new_sd, strict=True)
maxp, maxw, agr = compare(model)

if maxw > 0.1 or maxp > 0.1:
    # p_head.0 and v_head.0 have identical shapes -> try the one possible swap
    print("verification weak -- retrying with p/v head swap")
    swapped = list(pairs); swapped[13], swapped[14] = swapped[14], swapped[13]
    model.load_state_dict(build(swapped), strict=True)
    maxp, maxw, agr = compare(model)

print(f"max |d policy softmax| = {maxp:.4f} | max |d wdl softmax| = {maxw:.4f} "
      f"| policy top-1 agreement = {agr:.1%}")
assert maxw <= 0.1 and maxp <= 0.1 and agr > 0.95, "rebuild verification failed -- do not train on this"

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # fresh m/v
torch.save({"model": model.state_dict(),
            "ema": {k: v.clone() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(), "step": STEP,
            "best_loss": float("inf"), "wandb_run_id": None,
            "net_cfg": {"blocks": BLOCKS, "channels": CHANNELS}}, OUT)
print(f"wrote {OUT}")