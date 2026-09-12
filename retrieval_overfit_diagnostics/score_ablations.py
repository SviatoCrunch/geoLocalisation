"""§9 scoring ablations (no retrain): learned β vs uniform vs max, full-gallery retrieval."""
from __future__ import annotations

from pathlib import Path

import torch

from .common import rank_matrix, q_embed, write_json


@torch.no_grad()
def _split_ablations(ctx, sr):
    a = ctx.args
    qids = [q for q in sr.query_ids if ctx.store.has(q)]
    qid_row = {q: i for i, q in enumerate(sr.query_ids)}
    if not qids:
        return {"n": 0}
    Q = torch.stack([q_embed(ctx.model, ctx.store, q, ctx.device) for q in qids])
    det = ctx.model.score_with_details(Q, ctx.galV(), tile_chunk=a.eval_chunk)
    ls = det["level_scores"]                                   # (B, M, L)
    beta = det["beta"]                                         # (B, L)
    learned = (ls * beta[:, None, :]).sum(dim=2)               # == tile_scores
    uniform = ls.mean(dim=2)
    mx = ls.max(dim=2).values
    return {
        "n": len(qids),
        "learned_beta": rank_matrix(learned.cpu().numpy(), qids, qid_row, sr),
        "uniform_mean": rank_matrix(uniform.cpu().numpy(), qids, qid_row, sr),
        "max_level": rank_matrix(mx.cpu().numpy(), qids, qid_row, sr),
    }


def run(ctx) -> dict:
    out = {"note": "fixed-rule comparison; NO weights tuned on any labels. uniform/max > learned "
                   "signals a ScaleGate problem (not a proof of its cause).",
           "train": _split_ablations(ctx, ctx.sr["train"]),
           "val": _split_ablations(ctx, ctx.sr["val"])}
    write_json(Path(ctx.args.output_dir) / "ablation_metrics.json", out)
    return out
