"""Retrieval eval — Recall@K + median rank over a split's queries vs the whole gallery."""
from __future__ import annotations

import numpy as np
import torch


def build_gallery_V(model, tile_loader, tile_ids, tile_chunk, dev, progress=False):
    """All tiles → V (chunked to bound memory). Rows follow ``tile_ids`` order.

    ``tile_loader`` should be a NON-caching loader here: the full gallery (11465 tiles) must not be
    held in RAM. Progress bar because this pass streams every tile grid from disk each eval."""
    rng = range(0, len(tile_ids), tile_chunk)
    if progress:
        from tqdm import tqdm
        rng = tqdm(rng, desc="eval galV", unit="chunk", total=(len(tile_ids) + tile_chunk - 1) // tile_chunk)
    parts = None
    for s in rng:
        G = tile_loader.stack(tile_ids[s:s + tile_chunk]).float().to(dev)
        V = model.build_V(G)
        if parts is None:
            parts = {n: [] for n in V}
        for n in V:
            parts[n].append(V[n].detach())
        del G, V
    return {n: torch.cat(parts[n], 0) for n in parts}


@torch.no_grad()
def score_against_gallery(model, store, sr, galV, dev, tile_chunk=64, ks=(1, 5, 10, 20)):
    """Rank ``sr``'s queries against a PREBUILT ``galV`` → Recall@ks + median rank. Gallery-row
    order must match ``sr.tile_ids`` (== relevance pos rows). Lets val + test share one galV."""
    qids = [q for q in sr.query_ids if store.has(q)]
    if not qids:
        return {"n": 0, "median_rank": float("nan"), **{f"R@{k}": 0.0 for k in ks}}
    Q = torch.stack([model.encode_query(store.tokens(q).to(dev)) for q in qids])
    S = model.score(Q, galV, tile_chunk=tile_chunk)                 # (B, M)
    order = S.argsort(dim=1, descending=True).cpu().numpy()
    qid_row = {q: i for i, q in enumerate(sr.query_ids)}
    ranks, hit = [], {k: 0 for k in ks}
    for bi, q in enumerate(qids):
        pos = set(int(x) for x in sr.relevance.pos_of(qid_row[q]))
        if not pos:
            continue
        ranked = order[bi]
        r = next((j for j, t in enumerate(ranked) if int(t) in pos), len(ranked))
        ranks.append(r + 1)
        for k in ks:
            if r < k:
                hit[k] += 1
    # distance-based recall: is the TOP-1 tile geographically near the query GT? (fair on a
    # stride-250 gallery where the exact GT tile's neighbours are near-identical). EPSG:3857
    # euclidean → true metres via ~cos(lat) at these cities (~48.9°).
    _COS = 0.657
    dmeters, dthr = [], (250.0, 500.0, 1000.0)
    for bi, q in enumerate(qids):
        pos = set(int(x) for x in sr.relevance.pos_of(qid_row[q]))
        if not pos:
            continue
        top1 = int(order[bi][0])
        d = float(np.linalg.norm(sr.tile_xy[top1] - sr.q_xy[qid_row[q]])) * _COS
        dmeters.append(d)
    n = len(ranks)
    out = {"n": n, "median_rank": float(np.median(ranks)) if ranks else float("nan"),
           **{f"R@{k}": (hit[k] / n if n else 0.0) for k in ks}}
    if dmeters:
        dm = np.array(dmeters)
        out["median_dist_m"] = float(np.median(dm))
        out.update({f"distR@{int(t)}m": float((dm <= t).mean()) for t in dthr})
    return out


@torch.no_grad()
def evaluate(model, store, sr, tile_loader, dev, tile_chunk=64, ks=(1, 5, 10, 20), progress=False):
    """Convenience: build ``sr``'s gallery V then score it. (The trainer builds galV once and calls
    :func:`score_against_gallery` for both val and test — galV is identical across splits.)"""
    model.eval()
    galV = build_gallery_V(model, tile_loader, sr.tile_ids, tile_chunk, dev, progress=progress)
    return score_against_gallery(model, store, sr, galV, dev, tile_chunk, ks)
