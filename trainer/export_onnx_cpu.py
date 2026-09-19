#!/usr/bin/env python3
"""Export a ChessNet checkpoint to a CPU-optimized fp32 ONNX.

Why this exists
---------------
`export_onnx.py` produces a model with fp16 *internal* weights for GPU
inference. That is optimal on GPUs, but catastrophic on CPU: most server
CPUs (Modal's fleet included) do not have native fp16 arithmetic, so ONNX
Runtime's CPU execution provider inserts fp16->fp32 Cast nodes at every
layer boundary, roughly doubling inference cost.

This script is identical to export_onnx.py *except* it skips the final
fp16 conversion, leaving the graph in pure fp32. That is what the CPU EP
wants, and it makes CPU inference ~2x faster with no accuracy loss.

The input/output dtypes are unchanged from the original export:
    planes  : uint8   [batch, 113, 8]
    scalars : float32 [batch, 9]
    policy  : float32 [batch, 4672]
    wdl     : float32 [batch, 3]
    material: float32 [batch, 1]

So the C++ side (`engine/src/nn_client.cpp`) does NOT need any changes.
You can swap this ONNX in for the fp16 one and the engine will run it
without modification.

Usage
-----
    python trainer/export_onnx_cpu.py \
        --checkpoint checkpoints/latest.pt \
        --out models/champion_fp32.onnx

    # Optional: verify the export against PyTorch
    python trainer/export_onnx_cpu.py \
        --checkpoint checkpoints/latest.pt \
        --out models/champion_fp32.onnx \
        --verify
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.model import NUM_PLANES, NUM_SCALARS, build_model  # noqa: E402


class BitPackedInput(nn.Module):
    """Identical to the wrapper in export_onnx.py.

    Exposes the same uint8 `planes` input and unpacks each byte into 8 rank
    bits via a LUT, producing the [B, 113, 8, 8] float tensor the network
    expects. Kept byte-for-byte compatible so the C++ engine's existing
    `nn_client.cpp` works without modification.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        lut = torch.zeros(256, 8, dtype=torch.float32)
        for v in range(256):
            for i in range(8):
                lut[v, i] = float((v >> i) & 1)
        self.register_buffer("lut", lut, persistent=False)

    def forward(self, planes_u8: torch.Tensor, scalars: torch.Tensor):
        # planes_u8: [B, 113, 8] uint8
        # lut[planes_u8.long()] -> [B, 113, 8, 8] float32
        planes = self.lut[planes_u8.long()].reshape(-1, NUM_PLANES, 8, 8)
        return self.net(planes, scalars)


def _extract_state(ckpt_path: Path):
    """Load a trainer checkpoint and return (state_dict, net_cfg_or_None)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net_cfg = ck.get("net_cfg")
    if "ema" in ck and ck["ema"]:
        state, source = ck["ema"], "ema"
    elif "model" in ck:
        state, source = ck["model"], "model"
    else:
        state, source = ck, "raw"
    return state, net_cfg, source


def export(checkpoint: str | None, out: str, blocks: int, channels: int,
           batch_size: int, verify: bool) -> None:
    # ---- 1. Resolve architecture: checkpoint's net_cfg > CLI args ----------
    if checkpoint:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise SystemExit(f"checkpoint not found: {ckpt_path}")
        state, net_cfg, source = _extract_state(ckpt_path)
        if net_cfg and "blocks" in net_cfg and "channels" in net_cfg:
            blocks = net_cfg["blocks"]
            channels = net_cfg["channels"]
            print(f"[export] net_cfg from checkpoint: "
                  f"blocks={blocks} channels={channels}")
        else:
            print(f"[export] no net_cfg in checkpoint; using CLI: "
                  f"blocks={blocks} channels={channels}")
        print(f"[export] loaded weights from {ckpt_path} (source={source})")
    else:
        print(f"[export] no checkpoint; exporting randomly initialized net "
              f"(blocks={blocks} channels={channels})")
        state = None

    # ---- 2. Build model and load state ------------------------------------
    model = build_model({"blocks": blocks, "channels": channels})
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[export] WARNING: {len(missing)} missing keys, "
                  f"e.g. {missing[:3]}")
        if unexpected:
            print(f"[export] WARNING: {len(unexpected)} unexpected keys, "
                  f"e.g. {unexpected[:3]}")
    model.eval()

    # ---- 3. Wrap for bit-packed uint8 input --------------------------------
    wrapped = BitPackedInput(model)
    wrapped.eval()

    # ---- 4. Dummy inputs (uint8 planes, float32 scalars) ------------------
    planes  = torch.randint(0, 256, (batch_size, NUM_PLANES, 8), dtype=torch.uint8)
    scalars = torch.rand(batch_size, NUM_SCALARS, dtype=torch.float32)

    # ---- 5. Export ---------------------------------------------------------
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (planes, scalars),
        str(out_path),
        input_names=["planes", "scalars"],
        output_names=["policy", "wdl", "material"],
        dynamic_axes={
            "planes":   {0: "batch"},
            "scalars":  {0: "batch"},
            "policy":   {0: "batch"},
            "wdl":      {0: "batch"},
            "material": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"[export] wrote {out_path}")

    # ---- 6. Sanity check: inspect the graph inputs -------------------------
    try:
        import onnx
        m = onnx.load(str(out_path))
        print("[export] graph inputs:")
        for inp in m.graph.input:
            tt = inp.type.tensor_type
            dims = [d.dim_value if d.dim_value else d.dim_param
                    for d in tt.shape.dim]
            print(f"[export]   {inp.name}: elem_type={tt.elem_type} shape={dims}")
        print("[export] (elem_type: 1=FLOAT, 2=UINT8, 6=INT32, 7=INT64)")
        # Confirm no fp16 (elem_type 10) got introduced anywhere.
        import onnx.helper as oh
        fp16_types = {oh.TensorProto.FLOAT16}
        n_fp16_init = sum(1 for init in m.graph.initializer
                          if init.data_type in fp16_types)
        if n_fp16_init:
            print(f"[export] WARNING: {n_fp16_init} fp16 initializers present; "
                  f"CPU EP will insert casts")
        else:
            print("[export] no fp16 initializers; graph is pure fp32. Good.")
    except ImportError:
        print("[export] onnx package not installed; skipping graph inspection")

    # ---- 7. Optional parity check -----------------------------------------
    if verify:
        _verify(wrapped, planes, scalars, out_path)


def _verify(wrapped, planes, scalars, out_path: Path):
    """Run the exported ONNX through onnxruntime and compare to PyTorch."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify] onnxruntime not installed; skipping")
        return

    with torch.no_grad():
        p_t, w_t, m_t = wrapped(planes, scalars)
    p_t = p_t.numpy()
    w_t = w_t.numpy()
    m_t = m_t.numpy()

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(out_path), sess_options=so, providers=["CPUExecutionProvider"])

    # Feed the SAME uint8 planes and fp32 scalars the graph declares.
    p_np = planes.numpy().astype(np.uint8)
    s_np = scalars.numpy().astype(np.float32)

    p_o, w_o, m_o = sess.run(
        ["policy", "wdl", "material"],
        {"planes": p_np, "scalars": s_np},
    )

    def maxdiff(a, b):
        return float(np.abs(a - b).max())

    dp, dw, dm = maxdiff(p_t, p_o), maxdiff(w_t, w_o), maxdiff(m_t, m_o)
    print(f"[verify] max |Δ| policy={dp:.2e} wdl={dw:.2e} material={dm:.2e}")
    if max(dp, dw, dm) > 1e-3:
        print("[verify] *** FAIL: outputs diverge beyond 1e-3 ***")
        sys.exit(1)
    print("[verify] OK - exported model matches PyTorch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Export ChessNet to a pure-fp32 ONNX for CPU inference.")
    ap.add_argument("--checkpoint", default=None,
                    help="path to latest.pt (omit to export random init)")
    ap.add_argument("--out", required=True,
                    help="output ONNX path (e.g. models/champion_fp32.onnx)")
    ap.add_argument("--blocks", type=int, default=6,
                    help="only used if the checkpoint has no net_cfg")
    ap.add_argument("--channels", type=int, default=96,
                    help="only used if the checkpoint has no net_cfg")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="dummy batch size for export (default: 1)")
    ap.add_argument("--verify", action="store_true",
                    help="run parity check against onnxruntime after export")
    args = ap.parse_args()

    export(
        checkpoint=args.checkpoint,
        out=args.out,
        blocks=args.blocks,
        channels=args.channels,
        batch_size=args.batch_size,
        verify=args.verify,
    )