"""Shard dataset: loads engine output shards into training tensors.

Record layout mirrors trainer/shard_check.py docs (interleaved policy pairs).
"""
from __future__ import annotations

import gzip
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

NUM_PLANES = 113
POLICY_SLOTS = 64
SCALAR_COUNT = 9
SCALAR_BYTES = SCALAR_COUNT * 4
RECORD_BYTES = NUM_PLANES * 8 + SCALAR_BYTES + POLICY_SLOTS * 4 + 4
ACTIONS = 73 * 64

REC_DTYPE = np.dtype([
    ("planes", np.uint64, (NUM_PLANES,)),
    ("scalars", np.float32, (SCALAR_COUNT,)),
    ("pairs", np.uint16, (POLICY_SLOTS * 2,)),
    ("result", np.uint8),
    ("material_f16", np.uint16),
    ("stm", np.uint8),
])
assert REC_DTYPE.itemsize == RECORD_BYTES


def _f16_bits_to_float(bits):
    """IEEE half bit-pattern(s) -> float32."""
    b = np.asarray(bits, dtype=np.uint32)
    sign = np.where(b >> 15, -1.0, 1.0).astype(np.float32)
    exp = ((b >> 10) & 0x1F).astype(np.int32)
    man = (b & 0x3FF).astype(np.float32)
    val = np.where(exp == 0,
                   man * np.float32(2.0 ** -24),
                   (1.0 + man / 1024.0) * np.exp2((exp - 15).astype(np.float32)))
    return (sign * val).astype(np.float32)


def planes_to_tensor(words: np.ndarray) -> torch.Tensor:
    """[N,112] uint64 bit-words -> [N,112,8,8] float tensor (square s = bit s)."""
    n = words.shape[0]
    raw = np.ascontiguousarray(words).view(np.uint8).reshape(n, NUM_PLANES, 8)
    bits = np.unpackbits(raw, axis=2, bitorder="little")     # [n,112,64]
    return torch.from_numpy(bits.reshape(n, NUM_PLANES, 8, 8)).float()


class ShardDataset:
    """Dataset over engine shards with consumed-position ledger for exact resume.

    - refresh() never decompresses: position counts come from the 20-byte header
      plus the gzip ISIZE footer (uncompressed length mod 2^32).
    - Records are decompressed lazily into a path-keyed, byte-budgeted LRU cache.
    - Window trimming only evicts dropped files rather than flushing the entire cache.
    """

    def __init__(self, data_dir: str, cache_bytes: int = 3_000_000_000,
                 window_positions: int | None = None):
        self.data_dir = Path(data_dir)
        self.files: list[Path] = []
        self.counts: list[int] = []
        self.ledger: dict[str, int] = {}          # path -> positions consumed
        self._cache: dict[str, np.ndarray] = {}   # path(str) -> records
        self._cache_order: list[str] = []         # LRU list of paths (oldest first)
        self._cache_bytes = 0
        self.max_cache_bytes = cache_bytes
        self.window_positions = window_positions

    @staticmethod
    def _shard_positions(p: Path) -> int | None:
        """Read record count without decompressing the entire file."""
        try:
            # Check 20-byte header for magic bytes
            with gzip.open(p, "rb") as f:
                header = f.read(20)
            if len(header) < 20 or header[:4] != b"CBSP":
                return None

            # Read uncompressed size (ISIZE) from the last 4 bytes of the gzip file
            with open(p, "rb") as f:
                f.seek(-4, 2)  # 2 is os.SEEK_END
                (usize,) = struct.unpack("<I", f.read(4))

            usable = usize - 20
            if usable < 0 or usable % RECORD_BYTES != 0:
                return None
            return usable // RECORD_BYTES
        except Exception:
            return None

    def refresh(self) -> int:
        """Pick up newly completed shards using quick header/footer checks."""
        new_pos = 0
        known = {str(p) for p in self.files}
        for p in sorted(self.data_dir.rglob("*.gz")):
            sp = str(p)
            if sp in known or sp.endswith(".tmp"):
                continue

            n = self._shard_positions(p)
            if not n:
                continue

            self.files.append(p)
            self.counts.append(n)
            self.ledger[sp] = 0
            new_pos += n

        # Enforce replay window: keep NEWEST shards
        if self.window_positions:
            acc = 0
            cut = 0
            for i in range(len(self.files) - 1, -1, -1):
                acc += self.counts[i]
                if acc >= self.window_positions:
                    cut = i
                    break
            if cut > 0:
                dropped = self.files[:cut]
                print(f"[dataset] replay window: dropping {len(dropped)} old "
                      f"shards ({sum(self.counts[:cut])} positions)")

                # Selectively evict only the dropped files from the cache
                dropped_set = {str(p) for p in dropped}
                for sp in dropped_set:
                    arr = self._cache.pop(sp, None)
                    if arr is not None:
                        self._cache_bytes -= arr.nbytes
                self._cache_order = [sp for sp in self._cache_order if sp not in dropped_set]

                self.files = self.files[cut:]
                self.counts = self.counts[cut:]
        return new_pos

    def _load_array(self, p: Path) -> np.ndarray:
        with gzip.open(p, "rb") as f:
            data = f.read()
        usable = (len(data) - 20) // RECORD_BYTES * RECORD_BYTES
        return np.frombuffer(data[20:20 + usable], dtype=REC_DTYPE)

    def _records(self, shard_idx: int) -> np.ndarray:
        sp = str(self.files[shard_idx])
        arr = self._cache.get(sp)
        if arr is None:
            arr = self._load_array(self.files[shard_idx])
            self._cache[sp] = arr
            self._cache_order.append(sp)
            self._cache_bytes += arr.nbytes

            # Evict oldest shards until we are within byte budget
            while self._cache_bytes > self.max_cache_bytes and len(self._cache_order) > 1:
                old_sp = self._cache_order.pop(0)
                old_arr = self._cache.pop(old_sp, None)
                if old_arr is not None:
                    self._cache_bytes -= old_arr.nbytes
        return arr

    def prewarm(self, workers: int = 8) -> None:
        """Decompress window shards in parallel into RAM before training starts."""
        todo = [p for p in self.files if str(p) not in self._cache]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, arr in zip(todo, ex.map(self._load_array, todo)):
                sp = str(p)
                if sp in self._cache:
                    continue
                self._cache[sp] = arr
                self._cache_order.append(sp)
                self._cache_bytes += arr.nbytes

        while self._cache_bytes > self.max_cache_bytes and len(self._cache_order) > 1:
            old_sp = self._cache_order.pop(0)
            old_arr = self._cache.pop(old_sp, None)
            if old_arr is not None:
                self._cache_bytes -= old_arr.nbytes

    def mark_consumed(self, shard_idx: int, n: int):
        self.ledger[str(self.files[shard_idx])] += n

    def total_positions(self) -> int:
        return sum(self.counts)

    def __len__(self) -> int:
        return self.total_positions()

    def sample_batch(self, batch_size: int,
                     rng: np.random.Generator) -> dict[str, torch.Tensor]:
        """Uniformly sample a batch across all known positions."""
        if len(self.files) == 0:
            raise RuntimeError("no shards yet")
        counts = np.array(self.counts, dtype=np.int64)
        probs = counts / counts.sum()
        shard_choices = rng.choice(len(self.files), size=batch_size, p=probs)

        planes_b = np.empty((batch_size, NUM_PLANES), dtype=np.uint64)
        scalars_b = np.empty((batch_size, SCALAR_COUNT), dtype=np.float32)
        pol_b = np.zeros((batch_size, ACTIONS), dtype=np.float32)
        wdl_b = np.zeros((batch_size, 3), dtype=np.float32)
        mat_b = np.zeros(batch_size, dtype=np.float32)

        # group by shard for cache locality
        for si in np.unique(shard_choices):
            sel = np.where(shard_choices == si)[0]
            recs = self._records(int(si))
            offs = rng.integers(0, len(recs), size=len(sel))
            chunk = recs[offs]

            planes_b[sel] = chunk["planes"]
            scalars_b[sel] = chunk["scalars"]

            pairs = chunk["pairs"].reshape(len(sel), POLICY_SLOTS, 2)
            valid = pairs[:, :, 0] != 0xFFFF
            visits = np.where(valid, pairs[:, :, 1], 0).astype(np.float32)
            tot = visits.sum(axis=1, keepdims=True)
            tot[tot == 0] = 1.0
            dist = visits / tot

            # Vectorized scatter:
            sel_arr = np.asarray(sel, dtype=np.int64)
            cols = pairs[:, :, 0].astype(np.int64)
            flat_indices = sel_arr[:, None] * ACTIONS + cols
            pol_b.reshape(-1)[flat_indices[valid]] = dist[valid]

            results = chunk["result"].astype(np.int64)
            one_hot = np.zeros((len(sel), 3), dtype=np.float32)
            one_hot[np.arange(len(sel)), results] = 1.0
            stm_black = chunk["stm"].astype(bool)
            wdl = one_hot.copy()
            wdl[stm_black, 0], wdl[stm_black, 2] = (
                one_hot[stm_black, 2].copy(), one_hot[stm_black, 0].copy())
            wdl_b[sel] = wdl

            mat_b[sel] = _f16_bits_to_float(chunk["material_f16"])

        return {
            "planes": planes_to_tensor(planes_b),
            "scalars": torch.from_numpy(scalars_b.copy()),
            "policy_target": torch.from_numpy(pol_b),
            "wdl_target": torch.from_numpy(wdl_b),
            "material_target": torch.from_numpy(mat_b),
        }