"""§5 stale-gallery audit — is V built from the CURRENT checkpoint (not a cached/other one)?"""
from __future__ import annotations

from pathlib import Path

import torch

from .common import param_fingerprints, write_json, load_checkpoint, load_model

_TABLE = [
    {"representation": "raw DINO tokens", "frozen": True, "cacheable_across_ckpt": True,
     "path": "map_extract H5 ift_dino"},
    {"representation": "VLAD pre-projection", "frozen": "assign/centroids frozen",
     "cacheable_across_ckpt": True, "path": "aggregation.vlad._region / core.agg"},
    {"representation": "group projection output", "frozen": False, "cacheable_across_ckpt": False,
     "path": "core.group_proj (trainable)"},
    {"representation": "map head output", "frozen": False, "cacheable_across_ckpt": False,
     "path": "core.map_head (trainable)"},
    {"representation": "final pyramid V", "frozen": False, "cacheable_across_ckpt": False,
     "path": "geo_e2c_train.eval.build_gallery_V (rebuilt each eval)"},
]


def run(ctx) -> dict:
    a = ctx.args
    sr = ctx.sr["train"]
    tids = sr.tile_ids[:min(16, len(sr.tile_ids))]
    G = ctx.tiles.stack(tids).float().to(a.device)
    key = ctx.model.scales_cells[0]
    with torch.no_grad():
        V0 = ctx.model.build_V(G)[key]
        det = float((ctx.model.build_V(G)[key] - V0).abs().max())          # determinism
    # freshness proof: perturb a trainable map param → V must change; then restore. Use RANDOM
    # noise on the FINAL map_head linear — a uniform shift on an early layer is cancelled by the
    # head's LayerNorm, so it would falsely look "cached".
    changed = None
    mp = next((p for n, p in reversed(list(ctx.model.core.named_parameters()))
               if n.startswith("map_head") and n.endswith("weight") and p.requires_grad), None)
    if mp is not None:
        with torch.no_grad():
            orig = mp.detach().clone()
            mp.add_(torch.randn_like(mp) * 0.5)
            changed = float((ctx.model.build_V(G)[key] - V0).abs().max())
            mp.copy_(orig)
    # optional: compare to a second checkpoint
    ckb = None
    if a.checkpoint_b:
        mb, _ = load_model(load_checkpoint(a.checkpoint_b), device=a.device, assign_override=a.assign)
        with torch.no_grad():
            Vb = mb.build_V(G)[key]
        cos = torch.nn.functional.cosine_similarity(V0.reshape(-1), Vb.reshape(-1), dim=0)
        ckb = {"cosine": float(cos), "max_abs_diff": float((V0 - Vb).abs().max())}
    fresh = (det < 1e-4) and ((changed or 0) > 1e-4)
    out = {"param_fingerprints": param_fingerprints(ctx.model),
           "build_V_determinism_max_abs_diff": det,
           "V_changes_when_map_param_perturbed_max_abs_diff": changed,
           "checkpoint_b_diff": ckb, "representation_table": _TABLE,
           "eval_uses_current_checkpoint": bool(fresh),
           "note": "build_gallery_V rebuilds V from the loaded checkpoint each call (not cached across "
                   "checkpoints). determinism≈0 + perturbation>0 ⇒ V tracks the current map params."}
    write_json(Path(a.output_dir) / "cache_audit.json", out)
    return out
