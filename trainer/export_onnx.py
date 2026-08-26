"""Export a ChessNet checkpoint to fp16 ONNX for the C++ engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.model import NUM_PLANES, NUM_SCALARS, build_model  # noqa: E402


def export(checkpoint: str | None, out: str, blocks: int, channels: int,
           batch_size: int = 1) -> None:
    model = build_model({"blocks": blocks, "channels": channels})

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        print(f"loaded weights from {checkpoint}")
    else:
        print("no checkpoint given -> exporting randomly initialized net")

    model.eval()

    planes = torch.randint(0, 2, (batch_size, NUM_PLANES, 8, 8)).float()
    scalars = torch.rand(batch_size, NUM_SCALARS)

    torch.onnx.export(
        model,
        (planes, scalars),
        out,
        input_names=["planes", "scalars"],
        output_names=["policy", "wdl", "material"],
        dynamic_axes={
            "planes": {0: "batch"},
            "scalars": {0: "batch"},
            "policy": {0: "batch"},
            "wdl": {0: "batch"},
            "material": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"exported fp32 graph to {out}")

    # convert to mixed fp16 (keep io types stable)
    import onnx
    from onnxconverter_common import float16

    m = onnx.load(out)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, out)
    print(f"converted to fp16 internals: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--channels", type=int, default=160)
    args = ap.parse_args()
    export(args.checkpoint, args.out, args.blocks, args.channels)
