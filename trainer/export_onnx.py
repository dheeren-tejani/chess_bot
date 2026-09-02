"""Export a ChessNet checkpoint to fp16 ONNX for the C++ engine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.model import NUM_PLANES, NUM_SCALARS, build_model


class BitPackedInput(nn.Module):
    """Wrapper that unpacks [B, 113, 8] uint8 bytes into [B, 113, 8, 8] float planes on GPU."""
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        lut = torch.zeros(256, 8, dtype=torch.float32)
        for v in range(256):
            for i in range(8):
                lut[v, i] = float((v >> i) & 1)
        self.register_buffer("lut", lut, persistent=False)

    def forward(self, planes_u8: torch.Tensor, scalars: torch.Tensor):
        # planes_u8: [B, 113, 8] -> lut[planes_u8]: [B, 113, 8, 8]
        planes = self.lut[planes_u8.long()].reshape(-1, NUM_PLANES, 8, 8)
        return self.net(planes, scalars)


def export(checkpoint: str | None, out: str, blocks: int, channels: int,
           batch_size: int = 1) -> None:
    model = build_model({"blocks": blocks, "channels": channels})

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("ema", ckpt.get("model", ckpt))
        model.load_state_dict(state)
        print(f"loaded weights from {checkpoint} (EMA={ 'ema' in ckpt })")
    else:
        print("no checkpoint given -> exporting randomly initialized net")

    model.eval()
    wrapped_model = BitPackedInput(model)
    wrapped_model.eval()

    # Dynamic uint8 dummy tensor [batch, 113, 8]
    planes = torch.randint(0, 256, (batch_size, NUM_PLANES, 8), dtype=torch.uint8)
    scalars = torch.rand(batch_size, NUM_SCALARS, dtype=torch.float32)

    torch.onnx.export(
        wrapped_model,
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
    print(f"exported fp32 graph with bit-unpacking wrapper to {out}")

    # convert internal graph ops to fp16 (leaves uint8 and float32 inputs intact)
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