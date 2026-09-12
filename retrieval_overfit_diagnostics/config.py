"""CLI argument definition (shared by cli.py and tests)."""
from __future__ import annotations

import argparse

CHECKS = ["train-full-gallery", "inbatch-vs-full", "cache", "checkpoint", "labels",
          "gate", "ablations", "negatives", "split"]


def build_parser():
    ap = argparse.ArgumentParser("retrieval_overfit_diagnostics")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--checkpoint-b", default=None, help="second checkpoint for the freshness diff")
    ap.add_argument("--split-config", required=True)
    ap.add_argument("--split-json", required=True)
    ap.add_argument("--galleries", nargs="+", required=True, help="city=raw_map_h5")
    ap.add_argument("--queries", nargs="+", required=True, help="city=query_tokens_h5")
    ap.add_argument("--assign", default=None, help="override assign_path (else the checkpoint's)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checks", default="all", help="'all' or comma list: " + ",".join(CHECKS))
    ap.add_argument("--include-test", action="store_true",
                    help="also analyse TEST (never used for any selection; logged with a warning)")
    ap.add_argument("--allow-nonstrict", action="store_true",
                    help="load checkpoint with strict=False (the checkpoint audit still reports keys)")
    ap.add_argument("--query-size-m", type=float, default=1000.0)
    ap.add_argument("--safe-eps-area", type=float, default=0.0)
    ap.add_argument("--eval-chunk", type=int, default=64)
    ap.add_argument("--max-queries", type=int, default=0, help="cap queries per split (0=all; smoke)")
    ap.add_argument("--device", default="cuda")
    return ap


def resolve_checks(spec: str):
    if spec.strip() == "all":
        return list(CHECKS)
    out = [c.strip() for c in spec.split(",") if c.strip()]
    bad = [c for c in out if c not in CHECKS]
    if bad:
        raise SystemExit(f"unknown checks {bad}; allowed: {CHECKS}")
    return out
