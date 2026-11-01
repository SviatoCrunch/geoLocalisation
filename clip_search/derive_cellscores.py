"""Derive per-(frame,cell) cell scores + per-frame ranking from the full raw dump.

Reads each video's tile_scores_full.csv (frame,cell,pos,level_m,score,pos_lat,pos_lon) and writes
cell_scores_per_frame.csv (frame,rank,cell,cell_score,best_lat,best_lon), where
  pyramid(pos)   = Σ over levels of score,
  cell_score     = mean over the cell's positions of pyramid  (the prod cell-score definition),
  best_lat/lon   = the position with the max pyramid,
  rank           = per-frame rank of the cell by cell_score (1 = best).
No GPU; pure post-processing of the saved raw matrix. Idempotent.

Usage: python derive_cellscores.py <root_dir>
"""
import csv, glob, os, sys
from collections import defaultdict
from pathlib import Path


def derive(raw):
    out = str(Path(raw).parent / 'cell_scores_per_frame.csv')
    pyr = defaultdict(float)
    coord = {}
    with open(raw) as f:
        r = csv.reader(f)
        next(r)
        for frame, cell, pos, lvl, score, plat, plon in r:
            k = (frame, cell, pos)
            pyr[k] += float(score)
            coord[k] = (float(plat), float(plon))
    agg = defaultdict(list)
    for (frame, cell, pos), p in pyr.items():
        agg[(frame, cell)].append((p,) + coord[(frame, cell, pos)])
    byframe = defaultdict(list)
    for (frame, cell), lst in agg.items():
        cs = sum(x[0] for x in lst) / len(lst)
        bp = max(lst, key=lambda x: x[0])
        byframe[frame].append((cs, cell, bp[1], bp[2]))
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'rank', 'cell', 'cell_score', 'best_lat', 'best_lon'])
        for frame in sorted(byframe):
            for rank, (cs, cell, la, lo) in enumerate(sorted(byframe[frame], reverse=True), 1):
                w.writerow([frame, rank, cell, round(cs, 6), la, lo])
    return out


def main(root):
    n = 0
    for raw in sorted(glob.glob(os.path.join(root, '*', 'tile_scores_full.csv'))):
        if os.path.getsize(raw) == 0:
            continue
        derive(raw)
        n += 1
    print('derived cell_scores for', n, 'videos')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
