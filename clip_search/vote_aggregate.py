"""Video geo-localization by frame VOTING over direct km cells.

Input: per-video cell_scores_per_frame.csv (frame,rank,cell,cell_score,best_lat,best_lon)
where cell_score = mean over the cell's 25 pyramids (each pyramid = sum of 8 levels).

Scheme per video:
  1) per frame, cells are already ranked by cell_score;
  2) aggregate across frames by RECIPROCAL-RANK votes (scale-invariant): score(cell)=Σ_frames 1/rank;
  3) (optional) geographic NMS: when building the top-K, skip a cell within `nms_m` of an already
     chosen one, so the K slots cover K DISTINCT ~km areas (merges the near-GT signal that splits
     across adjacent direct cells);
  4) success = a top-K cell within D metres of GT.  Primary: top-10 @ 1000 m.

Usage: python vote_aggregate.py <root_dir> [method=rr] [nms_m=900]
"""
import csv, glob, math, os, re, sys
from collections import defaultdict


def hav(a, b, c, d):
    R = 6371000.0
    p1, p2 = math.radians(a), math.radians(c)
    dp, dl = math.radians(c - a), math.radians(d - b)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(min(1, math.sqrt(x)))


def gt_of(fr):
    f = [float(x) for x in re.findall(r'-?\d+\.\d+', fr)]
    return (f[0], f[1]) if len(f) >= 2 else None


def load(fp):
    byframe = defaultdict(list)
    gt = None
    with open(fp) as f:
        for row in csv.DictReader(f):
            gt = gt or gt_of(row['frame'])
            byframe[row['frame']].append((int(row['rank']), row['cell'], float(row['cell_score']),
                                          float(row['best_lat']), float(row['best_lon'])))
    return byframe, gt


def aggregate(byframe, method='rr'):
    """-> list of [cell, score, lat, lon] ranked desc. method: rr|votes|sum|mean."""
    acc = defaultdict(float)
    cnt = defaultdict(int)
    ll = {}
    for fr, rows in byframe.items():
        for rk, c, cs, la, lo in rows:
            if method == 'rr':
                acc[c] += 1.0 / rk
            elif method == 'votes':
                acc[c] += 1 if rk <= 10 else 0
            elif method == 'sum':
                acc[c] += cs
            elif method == 'mean':
                acc[c] += cs
                cnt[c] += 1
            if c not in ll or cs > ll[c][0]:
                ll[c] = (cs, la, lo)
    if method == 'mean':
        for c in acc:
            acc[c] /= max(1, cnt[c])
    return sorted(([c, s, ll[c][1], ll[c][2]] for c, s in acc.items()), key=lambda x: x[1], reverse=True)


def geo_nms(ranked, nms_m=900.0):
    out = []
    for c, s, la, lo in ranked:
        if all(hav(la, lo, p[2], p[3]) > nms_m for p in out):
            out.append([c, s, la, lo])
    return out


def eval_video(fp, method='rr', nms_m=900.0):
    byframe, gt = load(fp)
    if not gt or not byframe:
        return None
    ranked = aggregate(byframe, method)
    if nms_m:
        ranked = geo_nms(ranked, nms_m)
    dists = [hav(gt[0], gt[1], c[2], c[3]) for c in ranked]
    allmin = min(hav(gt[0], gt[1], la, lo) for rows in byframe.values() for _, _, _, la, lo in rows)
    return dists, allmin, len(byframe)


def main(root, method='rr', nms_m=900.0):
    KS = [1, 5, 10]
    DS = [250, 500, 1000]
    hit = {(k, d): 0 for k in KS for d in DS}
    ceil = {d: 0 for d in DS}
    n = 0
    rows = []
    for fp in sorted(glob.glob(os.path.join(root, '*', 'cell_scores_per_frame.csv'))):
        r = eval_video(fp, method, nms_m)
        if not r:
            continue
        dists, allmin, F = r
        n += 1
        v = os.path.basename(os.path.dirname(fp))[:8]
        for d in DS:
            if allmin <= d:
                ceil[d] += 1
        rows.append((v, F, round(min(dists[:10]) if dists else 9e9), round(allmin)))
        for k in KS:
            for d in DS:
                if any(x <= d for x in dists[:k]):
                    hit[(k, d)] += 1
    print('method=%s  nms_m=%s  videos=%d' % (method, nms_m, n))
    print('CEILING @250/500/1000: %d/%d/%d' % (ceil[250], ceil[500], ceil[1000]))
    print('%-6s %-8s %-8s %-8s' % ('', '@250', '@500', '@1000'))
    for k in KS:
        print('top%-3d %-8s %-8s %-8s' % (k, '%d/%d' % (hit[(k, 250)], n),
                                          '%d/%d' % (hit[(k, 500)], n), '%d/%d' % (hit[(k, 1000)], n)))
    print('\nper-video best-of-top10 (m) | ceiling_min:')
    for v, F, b, a in sorted(rows, key=lambda x: x[2]):
        print('  %-9s F=%-2d top10=%-7d ceil=%d' % (v, F, b, a))


if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    method = sys.argv[2] if len(sys.argv) > 2 else 'rr'
    nms = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
    main(root, method, nms)
