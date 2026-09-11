"""Retrieval eval — Recall@K + median rank over a split's queries vs the whole gallery."""
from __future__ import annotations

import numpy as np
import torch


def build_gallery_V(model, tile_loader, tile_ids, tile_chunk, dev):
    """All tiles → V (chunked to bound memory). Rows follow ``tile_ids`` order."""
    parts = None
    cuda = str(dev).startswith("cuda")
    for s in range(0, len(tile_ids), tile_chunk):
        G = tile_loader.stack(tile_ids[s:s + tile_chunk]).float().to(dev)
        V = model.build_V(G)
        if parts is None:
            parts = {n: [] for n in V}
        for n in V:
            parts[n].append(V[n].detach())
        del G, V
        if cuda:                                    # release the per-chunk residual cdist temp
            torch.cuda.empty_cache()
    return {n: torch.cat(parts[n], 0) for n in parts}


@torch.no_grad()
def evaluate(model, store, sr, tile_loader, dev, tile_chunk=256, ks=(1, 5, 10, 20)):
    """``sr`` = SplitRelevance (val/test). Returns Recall@ks + median rank over queries with a
    positive whose tokens exist. Gallery-row order == ``sr.tile_ids`` (== relevance pos rows)."""
    model.eval()
    galV = build_gallery_V(model, tile_loader, sr.tile_ids, tile_chunk, dev)
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
    n = len(ranks)
    return {"n": n, "median_rank": float(np.median(ranks)) if ranks else float("nan"),
            **{f"R@{k}": (hit[k] / n if n else 0.0) for k in ks}}
