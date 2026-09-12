"""§10 negative-mining audit — candidate size, mining staleness, hard-distractor coverage."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .common import q_embed, write_json


@torch.no_grad()
def run(ctx) -> dict:
    a = ctx.args
    sr = ctx.sr["train"]
    cand = [len(sr.relevance.pos_of(i)) + len(sr.relevance.safe_of(i)) for i in range(len(sr.query_ids))]

    # coverage: how many of each query's top-N full-gallery distractors are in its train candidate set
    qids = [q for q in sr.query_ids if ctx.store.has(q)]
    qid_row = {q: i for i, q in enumerate(sr.query_ids)}
    Q = torch.stack([q_embed(ctx.model, ctx.store, q, a.device) for q in qids])
    S = ctx.model.score(Q, ctx.galV(), tile_chunk=a.eval_chunk)
    order = S.argsort(dim=1, descending=True).cpu().numpy()
    Ns, cov = (10, 50, 100), {10: [], 50: [], 100: []}
    for bi, q in enumerate(qids):
        i = qid_row[q]
        candset = (set(int(x) for x in sr.relevance.pos_of(i))
                   | set(int(x) for x in sr.relevance.safe_of(i)))
        for N in Ns:
            topN = set(int(t) for t in order[bi][:N])
            cov[N].append(len(topN & candset) / N)

    out = {"candidate_count_per_query": {"median": int(np.median(cand)), "min": int(min(cand)),
                                         "max": int(max(cand)), "mean": float(np.mean(cand))},
           "mining_staleness": "geo_e2c_train.train builds the neighbour cache ONCE "
                               "(build_neighbour_cache(epoch=0)) from pre-training embeddings — "
                               "hard-negative selection is STATIC across all epochs.",
           "full_gallery_distractor_coverage_mean": {f"top{N}": float(np.mean(cov[N])) for N in Ns},
           "note": "low coverage ⇒ the hardest full-gallery distractors are largely ABSENT from the "
                   "train candidate set (the objective only sees easy negatives)."}
    write_json(Path(a.output_dir) / "negative_mining_audit.json", out)
    return out
