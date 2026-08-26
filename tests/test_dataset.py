"""Dataset round-trip + training smoke tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trainer.dataset import NUM_PLANES, ShardDataset, planes_to_tensor

SMOKE_DIR = Path(__file__).resolve().parent.parent / "data" / "verify"


@pytest.mark.skipif(not SMOKE_DIR.exists() or not list(SMOKE_DIR.glob("*.gz")),
                    reason="generate shards first (selfplay smoke run)")
def test_dataset_batch():
    ds = ShardDataset(str(SMOKE_DIR))
    ds.refresh()
    assert len(ds) > 0
    rng = np.random.default_rng(0)
    b = ds.sample_batch(8, rng)
    assert b["planes"].shape == (8, NUM_PLANES, 8, 8)
    assert b["planes"].dtype.is_floating_point
    assert set(b["planes"].flatten().tolist()).issubset({0.0, 1.0})
    ps = b["policy_target"].sum(dim=1)
    assert torch_allclose_one(ps)
    assert b["wdl_target"].sum(dim=1).min().item() == 1.0


def torch_allclose_one(t):
    return bool((t - 1.0).abs().max() < 1e-5)


def test_planes_roundtrip():
    from trainer.dataset import NUM_PLANES
    words = np.zeros(NUM_PLANES, dtype=np.uint64)
    words[0] = 0xDEADBEEFCAFEBABE
    words[1] = 1 << 63
    t = planes_to_tensor(words.reshape(1, NUM_PLANES))
    w = int(words[0])
    for s in range(64):
        expect = float((w >> s) & 1)
        got = t[0, 0, s // 8, s % 8].item()
        assert expect == got, f"sq {s}"
    assert t[0, 1, 7, 7].item() == 1.0   # bit 63 of word 1
