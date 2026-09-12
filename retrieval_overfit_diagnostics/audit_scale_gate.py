"""§8 ScaleGate diagnostics — β distribution, best-level definitions, gate-collapse flag."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .common import q_embed, write_json


@torch.no_grad()
def _stats(ctx, sr):
    a = ctx.args
    qids = [q for q in sr.query_ids if ctx.store.has(q)]
    if not qids:
        return {"n": 0}
    Q = torch.stack([q_embed(ctx.model, ctx.store, q, ctx.device) for q in qids])
    det = ctx.model.score_with_details(Q, ctx.galV(), tile_chunk=a.eval_chunk)
    beta = det["beta"].cpu().numpy()                          # (B, L) query-only
    lv = det["level_values"].cpu().numpy().tolist()
    L = beta.shape[1]
    ent = -(np.clip(beta, 1e-9, 1) * np.log(np.clip(beta, 1e-9, 1))).sum(1)
    gate_arg = beta.argmax(1)
    # best-levels at the TOP-1 retrieved tile per query
    tile_scores = det["tile_scores"].cpu().numpy()
    top1 = tile_scores.argmax(1)
    sim = det["similarity_best_level_index"].cpu().numpy()[np.arange(len(qids)), top1]
    con = det["contribution_best_level_index"].cpu().numpy()[np.arange(len(qids)), top1]
    disagree = float(np.mean((sim != gate_arg) | (con != gate_arg) | (sim != con)))
    hist = lambda arr: {int(k): int(v) for k, v in sorted(Counter(arr.tolist()).items())}
    dom_level, dom_frac = Counter(gate_arg.tolist()).most_common(1)[0]
    return {
        "n": len(qids), "n_levels": L, "level_values": lv,
        "beta_mean": beta.mean(0).tolist(), "beta_std": beta.std(0).tolist(),
        "gate_argmax_hist": hist(gate_arg), "similarity_best_hist": hist(sim),
        "contribution_best_hist": hist(con),
        "entropy_mean": float(ent.mean()), "entropy_min": float(ent.min()),
        "three_defs_disagree_frac": disagree,
        "dominant_gate_level": int(dom_level), "dominant_gate_frac": float(dom_frac / len(qids)),
        "potential_gate_collapse": bool(dom_frac / len(qids) > 0.9),
    }


def run(ctx) -> dict:
    out = {"train": _stats(ctx, ctx.sr["train"]), "val": _stats(ctx, ctx.sr["val"]),
           "note": "potential_gate_collapse is a flag (dominant level >90%), NOT a proven defect — "
                   "compare against a ground-truth footprint distribution before concluding."}
    write_json(Path(ctx.args.output_dir) / "scale_gate_summary.json", out)
    return out
