"""§4 decisive test — train candidate-set metric vs train full-gallery vs val full-gallery."""
from __future__ import annotations

from pathlib import Path

from .common import candidate_set_metrics, full_gallery_per_query, write_json


def _lvl(m, key="R@10"):
    v = m.get(key)
    return v if v is not None else 0.0


def interpret(obj, tr_full, val_full):
    """Coarse suspicion (NOT a verdict) from the §4 table. High/low on R@10."""
    def hi(m):
        return _lvl(m) >= 0.5
    def lo(m):
        return _lvl(m) < 0.1
    if not hi(obj) and lo(tr_full):
        s = "objective/gallery mismatch or weak negatives (train easy on its candidate set, hard on full gallery)"
    elif hi(tr_full) and lo(val_full):
        s = "memorization or split/domain shift (train full-gallery good, val poor)"
    elif lo(tr_full) and lo(val_full):
        s = "evaluation / cache / labels / checkpoint bug (near-zero loss but train full-gallery also low)"
    else:
        s = "some generalization present — inspect trend across epochs before concluding"
    return {"primary_suspect": s,
            "objective_R@10": _lvl(obj), "train_full_R@10": _lvl(tr_full), "val_full_R@10": _lvl(val_full)}


def run(ctx) -> dict:
    a = ctx.args
    out = Path(a.output_dir)
    tr, va = ctx.sr["train"], ctx.sr["val"]
    obj = candidate_set_metrics(ctx.model, ctx.store, tr, ctx.tiles, ctx.device, a.eval_chunk,
                                cand_cap=a.objective_cap, progress=True, max_queries=a.max_queries)
    galV = ctx.galV()
    tr_rows, tr_full = full_gallery_per_query(ctx.model, ctx.store, tr, galV, ctx.device, a.eval_chunk)
    va_rows, va_full = full_gallery_per_query(ctx.model, ctx.store, va, galV, ctx.device, a.eval_chunk)
    interp = interpret(obj, tr_full, va_full)
    write_json(out / "train_objective_metrics.json", {"train_objective": obj})
    write_json(out / "train_full_gallery_metrics.json", {"train_full_gallery": tr_full})
    write_json(out / "val_full_gallery_metrics.json", {"val_full_gallery": va_full})
    write_json(out / "decisive_summary.json",
               {"train_objective": obj, "train_full_gallery": tr_full,
                "val_full_gallery": va_full, "interpretation": interp})
    return {"train_objective": obj, "train_full_gallery": tr_full, "val_full_gallery": va_full,
            "interpretation": interp, "_per_query": {"train": tr_rows, "val": va_rows}}
