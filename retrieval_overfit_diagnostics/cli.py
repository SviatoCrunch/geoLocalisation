"""Single entry point — run selected checks, write outputs/<run_id>/ + a hypothesis report."""
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _write_per_query_csv(path, per_query: dict):
    rows = []
    for split, items in per_query.items():
        for r in items:
            rows.append({"group": split, **r})
    cols = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv=None) -> int:
    from .config import build_parser, resolve_checks
    from . import (common, eval_train_full_gallery, score_ablations, audit_scale_gate,
                   audit_positive_labels, audit_gallery_freshness, audit_checkpoint_and_modes,
                   audit_negatives, audit_split_shift, build_report)
    args = build_parser().parse_args(argv)
    checks = resolve_checks(args.checks)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("w", encoding="utf-8")

    def say(msg):
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    if args.include_test:
        say("[WARN] --include-test: TEST analysed for REPORTING ONLY; never used for any selection.")
    say(f"[cfg] checks={checks} ckpt={args.checkpoint}")

    ctx = common.build_context(args)
    common.write_json(out / "resolved_config.json", {"model": ctx.model_info, "args": vars(args)})
    say(f"[model] epoch={ctx.model_info.get('epoch')} agg={ctx.model_info['agg']} "
        f"pyramid={ctx.model_info['pyramid_mode']} d_token={ctx.model_info['d_token']}")

    reg = {"train-full-gallery": eval_train_full_gallery, "inbatch-vs-full": eval_train_full_gallery,
           "ablations": score_ablations, "gate": audit_scale_gate, "labels": audit_positive_labels,
           "cache": audit_gallery_freshness, "checkpoint": audit_checkpoint_and_modes,
           "negatives": audit_negatives, "split": audit_split_shift}

    results, per_query = {}, {}
    for c in checks:
        if c == "inbatch-vs-full" and "train-full-gallery" in results:   # same decisive test
            results[c] = results["train-full-gallery"]; say(f"[ok] check {c} (alias)"); continue
        mod = reg.get(c)
        if mod is None:
            continue
        try:
            res = mod.run(ctx)
        except Exception as e:                          # a failing audit must not sink the whole run
            say(f"[FAIL] check {c}: {type(e).__name__}: {e}")
            results[c] = {"error": f"{type(e).__name__}: {e}"}
            continue
        for k, v in (res.pop("_per_query", {}) or {}).items():
            per_query.setdefault(k, v)
        results[c] = res
        say(f"[ok] check {c}")

    _write_per_query_csv(out / "per_query.csv", per_query)
    build_report.run(results, out)
    say(f"[ok] diagnostics -> {out}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
