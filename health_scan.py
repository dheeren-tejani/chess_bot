"""Comprehensive dataset health scan - every shard, every record.

Usage: python health_scan.py [data_dir]   (default: data/selfplay)

The two collapse detectors:
  - draw rate per iteration (rising = spiral forming)
  - top-1 visit share (-> 1.0 means MCTS is near-deterministic; the first
    run's collapsed data sat at 0.92-0.93 with ~1.9 moves/position)
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
from trainer.dataset import REC_DTYPE

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/selfplay')
total_pos = 0
results = np.zeros(3, dtype=np.int64)
stm_counts = np.zeros(2, dtype=np.int64)
visit_sums = []
top1_shares = []
bad_shards = []
per_iter = {}

for p in sorted(root.rglob('*.gz')):
    try:
        import gzip
        data = gzip.open(p, 'rb').read()
        usable = (len(data) - 20) // REC_DTYPE.itemsize * REC_DTYPE.itemsize
        arr = np.frombuffer(data[20:20 + usable], dtype=REC_DTYPE)
        if len(arr) == 0 or arr['result'].max() > 2 or arr['stm'].max() > 1:
            bad_shards.append(p.name)
            continue
        n = len(arr)
        total_pos += n
        results += np.bincount(arr['result'], minlength=3)
        stm_counts += np.bincount(arr['stm'], minlength=2)
        pairs = arr['pairs'].reshape(n, 64, 2)
        valid = pairs[:, :, 0] != 0xFFFF
        vis = np.where(valid, pairs[:, :, 1], 0)
        tot = vis.sum(1)
        top = vis.max(1)
        visit_sums.append(tot.mean())
        top1_shares.append((top / np.maximum(tot, 1)).mean())
        itdir = p.parent.name
        per_iter.setdefault(itdir, [0, 0, 0.0, 0.0])
        per_iter[itdir][0] += n
        per_iter[itdir][1] += int((arr['result'] == 1).sum())
        per_iter[itdir][2] += float((top / np.maximum(tot, 1)).sum())
        per_iter[itdir][3] += float(tot.sum())
    except Exception as e:
        bad_shards.append(f"{p.name}: {e}")

print(f"dataset: {root}")
print(f"total positions: {total_pos:,}")
print(f"corrupt/bad shards: {len(bad_shards)} {bad_shards[:3]}")
print(f"outcome split: W={results[0]:,} D={results[1]:,} L={results[2]:,} "
      f"(draws={results[1]/max(total_pos,1)*100:.1f}%)")
print(f"stm balance: white={stm_counts[0]:,} black={stm_counts[1]:,}")
print(f"avg root visits per position: {np.mean(visit_sums):.0f}")
print(f"avg top-1 visit share: {np.mean(top1_shares):.3f} "
      f"(healthy < ~0.6; the collapsed run sat at 0.92+)")
print("\nper-directory (positions / draw-rate / top1-share / visits-per-pos):")
for k in sorted(per_iter):
    pos, dr, t1s, vs = per_iter[k]
    print(f"  {k}: {pos:>8,} pos, draws={dr/max(pos,1)*100:5.1f}%, "
          f"top1={t1s/max(pos,1):.3f}, visits/pos={vs/max(pos,1):.0f}")

draw_frac = results[1] / max(total_pos, 1)
if draw_frac > 0.90:
    print("\n*** VERDICT: draw rate > 90% - draw spiral. Check use_twofold_draw, "
          "max_plies vs adjudication, prior_plies, and consider re-genesis. ***")
elif np.mean(top1_shares) > 0.8:
    print("\n*** VERDICT: policy near-deterministic (top-1 share > 0.8) - "
          "exploration is dying even if draw rate looks OK. ***")
else:
    print("\nVERDICT: dataset looks healthy.")
