"""Stage-C search over the S3 DINOv3-sat per-cell pyramid store.

Per query (drone frame, DINOv3-sat tokens): take the coarse (DINOv2) shortlist's top-``k_coarse``
candidate cells, fetch each cell's pyramids from S3 by index (``CellStoreS3``), score every
(position, level) grid against the query with the GPU homography verifier (``gpu_verify``), then:

  * pyramid (position) score = MAX over its levels  (best level);
  * cell score = MEAN over its pyramids  (``--cell-agg`` also allows min/max);
  * return the top-``topk`` cells; inside the winning cell the predicted point is the pyramid with the
    highest score, at its best level.

Writes a JSON of ranked cells + a KMZ (GT + top-K points, each with a square of its level footprint).
Reports fetch / score / total timings for the speed assessment. Score formula = inliers /
n_query_patches (unchanged); only the level→pyramid→cell aggregation is the new max/mean rule.

Run::

    HF_TOKEN=… uv run --python 3.11 --with "torch==2.5.1" --with kornia --with h5py --with numpy \
      --with boto3 --with tqdm python -m patch_rerank.search_pyramid_s3 \
      --queries kup=/…/query_kup_dinov3sat.h5 \
      --shortlist /…/shortlist_kup_prod_k100.json \
      --index-uri s3://geo-reference/embeddings/kup/pyramid_dinov3sat/kup/_index.json \
      --k-coarse 100 --topk 5 --cell-agg mean --device cuda \
      --out /…/search_kup.json --kmz /…/search_kup.kmz
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _score_cell(cd, qfeat, qxy, n_q, args, device):
    """→ (pyramid_score[n_pos], best_level[n_pos]). All grids share the token layout (output_px/patch),
    so the whole cell is one batched mutual-NN + one batched homography verify."""
    import torch
    from .matcher import _MIN_MATCHES, grid_keypoints, matched_coords_batch_gpu
    from .gpu_verify import ransac_homography_batch

    keys, grids, kp = [], [], None
    for i in range(cd.n_pos):
        for L in cd.levels:
            g = cd.grid(i, L)
            h, w, D = g.shape
            if kp is None:
                kp = grid_keypoints(h, w)
            grids.append(g.reshape(h * w, D).to(device)); keys.append((i, L))
    pairs = matched_coords_batch_gpu(qfeat, torch.stack(grids), qxy, kp, device)
    inls = ransac_homography_batch([(qm, rm) for (qm, rm, _) in pairs], reproj_thresh=args.reproj_thresh,
                                   n_hyp=args.gpu_n_hyp, score_type=args.gpu_score_type,
                                   refine=args.gpu_refine_iter > 0, refine_iter=args.gpu_refine_iter,
                                   device=device, seed=args.gpu_seed)
    sc = {}
    for idx, (i, L) in enumerate(keys):
        n_mut = pairs[idx][2]
        sc[(i, L)] = (inls[idx].shape[0] / n_q) if (n_q and n_mut >= _MIN_MATCHES["homography"]) else 0.0
    pyr, blv = [], []
    for i in range(cd.n_pos):
        per = {L: sc[(i, L)] for L in cd.levels}
        bL = max(per, key=per.get)
        pyr.append(per[bL]); blv.append(int(bL))
    return pyr, blv


def _aggregate(pyr, blv, cd, cell_agg):
    """pyramid scores → cell score (mean|min|max) + best pyramid (point) & its level."""
    arr = np.asarray(pyr, float)
    cs = {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}[cell_agg]
    bi = int(arr.argmax())                                    # point = best pyramid in the cell
    return cs, bi, float(cd.lat[bi]), float(cd.lon[bi]), int(blv[bi])


def _square(lat, lon, side_m):
    """4-corner (closed) polygon coord string for a side_m×side_m box centred at lat/lon."""
    hlat = (side_m / 2) / 110540.0
    hlon = (side_m / 2) / (math.cos(math.radians(lat)) * 111320.0)
    c = [(lon - hlon, lat - hlat), (lon + hlon, lat - hlat), (lon + hlon, lat + hlat),
         (lon - hlon, lat + hlat), (lon - hlon, lat - hlat)]
    return " ".join(f"{x:.7f},{y:.7f}" for x, y in c)


def _write_kmz(path, per_query):
    head = ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">'
            '<Document>'
            '<Style id="gt"><IconStyle><color>ff00ff00</color></IconStyle></Style>'
            '<Style id="p1"><IconStyle><color>ff0000ff</color></IconStyle></Style>'
            '<Style id="pk"><IconStyle><color>ff00a5ff</color><scale>0.8</scale></IconStyle></Style>'
            '<Style id="sq"><LineStyle><color>ff00a5ff</color><width>2</width></LineStyle>'
            '<PolyStyle><fill>0</fill></PolyStyle></Style>')
    body = [head]
    for q, rec in per_query.items():
        d = rec.get("fine_dist_m")
        body.append(f"<Folder><name>{q}" + (f" ({d:.0f} m)" if d is not None else "") + "</name>")
        if rec.get("gt"):
            gl, go = rec["gt"]
            body.append(f'<Placemark><name>GT</name><styleUrl>#gt</styleUrl>'
                        f'<Point><coordinates>{go:.7f},{gl:.7f}</coordinates></Point></Placemark>')
        for t in rec["topk"]:
            st = "p1" if t["rank"] == 1 else "pk"
            nm = f"#{t['rank']} L{t['level_m']} s{t['cell_score']:.3f}"
            body.append(f'<Placemark><name>{nm}</name><styleUrl>#{st}</styleUrl>'
                        f'<Point><coordinates>{t["lon"]:.7f},{t["lat"]:.7f}</coordinates></Point></Placemark>')
            body.append(f'<Placemark><name>{nm} box</name><styleUrl>#sq</styleUrl>'
                        f'<Polygon><outerBoundaryIs><LinearRing><coordinates>'
                        f'{_square(t["lat"], t["lon"], t["level_m"])}'
                        f'</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
        body.append("</Folder>")
    body.append("</Document></kml>\n")
    p = Path(path).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("doc.kml", "".join(body))


def _report(name, dists):
    d = np.array(dists) if dists else np.array([np.nan])
    thr = (250.0, 500.0, 1000.0)
    r = {f"distR@{int(t)}m": float((d <= t).mean()) for t in thr}
    print(f"[{name}] n={len(dists)} median_dist_m={float(np.median(d)):.1f} "
          + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
    return {"n": len(dists), "median_dist_m": float(np.median(d)), **r}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", nargs="+", required=True, help="city=query_dinov3sat.h5 (DINOv3 tokens)")
    ap.add_argument("--shortlist", required=True, help="coarse (DINOv2) top-N cells per query, matching store ids")
    ap.add_argument("--index-uri", required=True, help="s3://…/<city>/_index.json manifest")
    ap.add_argument("--cache-dir", default="/tmp/cellcache")
    ap.add_argument("--cache-cap", type=int, default=256)
    ap.add_argument("--k-coarse", type=int, default=100, help="candidate cells refined per query")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--cell-agg", choices=["mean", "min", "max"], default="mean",
                    help="cell score over its pyramids (spec: mean)")
    ap.add_argument("--reproj-thresh", type=float, default=2.0)
    ap.add_argument("--gpu-n-hyp", type=int, default=256)
    ap.add_argument("--gpu-score-type", choices=["msac", "ransac"], default="msac")
    ap.add_argument("--gpu-refine-iter", type=int, default=5)
    ap.add_argument("--gpu-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-city", default=None)
    ap.add_argument("--day-only", action="store_true")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kmz", default=None)
    args = ap.parse_args(argv)

    import torch
    from tqdm import tqdm
    from .cell_store_s3 import CellStoreS3
    from .query_io import QueryGridStore

    qpaths = dict(a.split("=", 1) for a in args.queries)
    qstore = QueryGridStore(qpaths)
    store = CellStoreS3(args.index_uri, cache_dir=args.cache_dir, cache_cap=args.cache_cap)
    sj = json.loads(Path(args.shortlist).expanduser().read_text())
    dev = args.device

    out = {"meta": {"k_coarse": args.k_coarse, "topk": args.topk, "cell_agg": args.cell_agg,
                    "index_uri": args.index_uri, "store_config": store.config,
                    "gpu": {"n_hyp": args.gpu_n_hyp, "score_type": args.gpu_score_type,
                            "refine_iter": args.gpu_refine_iter}},
           "per_query": {}}
    per_query, d_fine, d_ftop, t_fetch_all, t_score_all = {}, [], [], 0.0, 0.0

    for q, entry in tqdm(list(sj["shortlist"].items()), desc="search", unit="q"):
        if args.only_city and q.split(":", 1)[0] != args.only_city:
            continue
        if args.day_only and "_night" in q:
            continue
        if not qstore.has(q):
            continue
        if args.max_queries and len(per_query) >= args.max_queries:
            break
        qfeat, qxy, qlat, qlon = qstore.get(q)
        qfeat = qfeat.to(dev); n_q = int(qfeat.shape[0])
        cands = [c for c in entry["cells"][:args.k_coarse] if store.has(c)]
        recs, tf, ts = [], 0.0, 0.0
        for cid in cands:
            t0 = time.perf_counter(); cd = store.cell(cid); tf += time.perf_counter() - t0
            t1 = time.perf_counter(); pyr, blv = _score_cell(cd, qfeat, qxy, n_q, args, dev)
            ts += time.perf_counter() - t1
            cs, bi, blat, blon, bL = _aggregate(pyr, blv, cd, args.cell_agg)
            recs.append({"cell_id": cid, "cell_score": cs, "lat": blat, "lon": blon, "level_m": bL})
        ranked = sorted(recs, key=lambda r: r["cell_score"], reverse=True)[:args.topk]
        has_gt = math.isfinite(qlat) and math.isfinite(qlon)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
            r["dist_m"] = _haversine_m(qlat, qlon, r["lat"], r["lon"]) if has_gt else None
        df = ranked[0]["dist_m"] if (ranked and has_gt) else None
        dft = min((r["dist_m"] for r in ranked), default=None) if has_gt else None
        rec = {"gt": [qlat, qlon] if has_gt else None, "fine_dist_m": df,
               "fine_dist_topk_m": dft, "n_candidates": len(cands), "topk": ranked,
               "timings": {"fetch_s": tf, "score_s": ts, "downloads": store.n_downloads}}
        per_query[q] = rec; out["per_query"][q] = rec
        t_fetch_all += tf; t_score_all += ts
        if has_gt:
            d_fine.append(df); d_ftop.append(dft)
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()

    if d_fine:
        out["summary"] = {"fine_top1": _report("fine", d_fine),
                          f"fine_top{args.topk}": _report(f"fine@top{args.topk}", d_ftop)}
    out["speed"] = {"queries": len(per_query), "fetch_s_total": t_fetch_all, "score_s_total": t_score_all,
                    "s_per_query": (t_fetch_all + t_score_all) / max(1, len(per_query)),
                    "cells_downloaded": store.n_downloads}
    print(f"[speed] queries={len(per_query)} fetch={t_fetch_all:.1f}s score={t_score_all:.1f}s "
          f"| {(t_fetch_all + t_score_all) / max(1, len(per_query)):.1f}s/q "
          f"downloads={store.n_downloads}", flush=True)
    Path(args.out).expanduser().write_text(json.dumps(out), encoding="utf-8")
    print(f"[ok] -> {args.out}", flush=True)
    if args.kmz:
        _write_kmz(args.kmz, per_query)
        print(f"[ok] kmz -> {args.kmz}", flush=True)
    store.close(); qstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
