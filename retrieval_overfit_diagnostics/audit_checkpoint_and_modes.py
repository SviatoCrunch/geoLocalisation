"""§6 checkpoint + train/eval mode audit — strict load, key match, eval determinism, dropout."""
from __future__ import annotations

from pathlib import Path

import torch

from .common import q_embed, write_json


def run(ctx) -> dict:
    a, info = ctx.args, ctx.model_info
    sr = ctx.sr["train"]
    tids = sr.tile_ids[:min(8, len(sr.tile_ids))]
    G = ctx.tiles.stack(tids).float().to(a.device)
    qid = next((q for q in sr.query_ids if ctx.store.has(q)), None)
    Q = q_embed(ctx.model, ctx.store, qid, a.device).unsqueeze(0)

    ctx.model.eval()
    with torch.no_grad():
        s1 = ctx.model.score(Q, ctx.model.build_V(G))
        s2 = ctx.model.score(Q, ctx.model.build_V(G))
    eval_det = float((s1 - s2).abs().max())

    ctx.model.train()
    with torch.no_grad():
        t1 = ctx.model.score(Q, ctx.model.build_V(G))
        t2 = ctx.model.score(Q, ctx.model.build_V(G))
    train_diff = float((t1 - t2).abs().max())
    ctx.model.eval()

    dropouts = [m for m in ctx.model.core.modules() if m.__class__.__name__ == "Dropout"]
    out = {"epoch": info.get("epoch"), "strict_load": not a.allow_nonstrict,
           "missing_keys": info["missing_keys"], "unexpected_keys": info["unexpected_keys"],
           "config": {"agg": info["agg"], "pyramid_mode": info["pyramid_mode"],
                      "k": info["k"], "d_token": info["d_token"]},
           "eval_determinism_max_abs_diff": eval_det,
           "train_mode_repeat_max_abs_diff": train_diff,
           "n_dropout_modules": len(dropouts),
           "any_dropout_training_in_eval": any(bool(m.training) for m in dropouts),
           "note": "eval repeats must match (~0); train repeats may differ (dropout active). No dropout "
                   "should be in training mode during eval; missing/unexpected keys must be empty."}
    write_json(Path(a.output_dir) / "checkpoint_audit.json", out)
    return out
