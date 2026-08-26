"""Validate binary shard files produced by the engine.

Shard record layout (little-endian, interleaved policy pairs):
    planes  : uint64[112]                       896 B
    scalars : float32[4]                         16 B
    policy  : {u16 action_idx, u16 visits} x 64 256 B   (idx 0xFFFF = unused slot)
    meta    : u8 result, u16 material_f16, u8 pad 4 B
"""
import gzip
import struct
import sys
from pathlib import Path
import numpy as np

NUM_PLANES = 113
POLICY_SLOTS = 64
SCALAR_COUNT = 9
RECORD_BYTES = NUM_PLANES * 8 + SCALAR_COUNT * 4 + POLICY_SLOTS * 4 + 4

REC_DTYPE = np.dtype([
    ("planes", np.uint64, (NUM_PLANES,)),
    ("scalars", np.float32, (SCALAR_COUNT,)),
    ("pairs", np.uint16, (POLICY_SLOTS * 2,)),   # interleaved idx,visits
    ("result", np.uint8),
    ("material_f16", np.uint16),
    ("stm", np.uint8),
])
assert REC_DTYPE.itemsize == RECORD_BYTES, (REC_DTYPE.itemsize, RECORD_BYTES)


def read_shard(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = data[:4]
    assert magic == b"CBSP", f"bad magic {magic}"
    version, num_games = struct.unpack("<II", data[4:12])
    model_hash, = struct.unpack("<Q", data[12:20])
    body = np.frombuffer(data[20:], dtype=REC_DTYPE)
    return version, num_games, model_hash, body


def policy_of(rec):
    """Return (idx[], visits[]) arrays of used slots."""
    p = rec["pairs"].reshape(POLICY_SLOTS, 2)
    mask = p[:, 0] != 0xFFFF
    return p[mask, 0], p[mask, 1]


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/smoke8")
    for p in sorted(root.glob("*.gz")):
        v, ng, mh, body = read_shard(p)
        totals = []
        mass_frac = []
        for rec in body:
            _, vis = policy_of(rec)
            tot = int(vis.astype(np.int64).sum())
            totals.append(tot)
            # fraction of mass inside top move (sanity: should often be sizable)
            if len(vis):
                mass_frac.append(int(vis.max()) / max(tot, 1))
        totals = np.array(totals)
        results = np.bincount(body["result"], minlength=3)
        pieces = sum(bin(int(w)).count("1") for w in body[0]["planes"][:12])
        print(f"{p.name}: v{v} games={ng} pos={len(body)} "
              f"W/D/L={results.tolist()} visits[min/med/max]="
              f"{totals.min()}/{int(np.median(totals))}/{totals.max()} "
              f"top-move-frac[med]={np.median(mass_frac):.2f} rec0pieces={pieces}")
