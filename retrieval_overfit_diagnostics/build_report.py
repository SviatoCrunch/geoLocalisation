"""§12 report — a hypothesis table (Supported/Refuted/Insufficient + evidence) from the audits.

Statuses are derived from metrics, not impressions. Nothing is forced to a single cause.
"""
from __future__ import annotations

from pathlib import Path

from .common import write_json


def _r10(m):
    return None if not m else m.get("R@10")


def run(results: dict, out_dir) -> dict:
    dec = results.get("train-full-gallery", {})
    obj, trf, vaf = dec.get("train_objective"), dec.get("train_full_gallery"), dec.get("val_full_gallery")
    cache, ck, gate = results.get("cache"), results.get("checkpoint"), results.get("gate")
    abl, neg = results.get("ablations"), results.get("negatives")
    H = []

    if cache is not None:
        fresh = (cache.get("build_V_determinism_max_abs_diff", 1) < 1e-4
                 and (cache.get("V_changes_when_map_param_perturbed_max_abs_diff") or 0) > 1e-4)
        H.append(("Eval uses current-checkpoint descriptors (not stale)",
                  "Supported" if fresh else "Insufficient",
                  f"determinism={cache.get('build_V_determinism_max_abs_diff')}, "
                  f"perturbΔ={cache.get('V_changes_when_map_param_perturbed_max_abs_diff')}"))
    if trf is not None:
        H.append(("High Recall on TRAIN queries over the FULL gallery",
                  "Supported" if (_r10(trf) or 0) >= 0.5 else "Refuted",
                  f"train_full R@10={_r10(trf)}, median_rank={trf.get('median_rank')}, "
                  f"distR@500m={trf.get('distR@500m')}"))
    if obj is not None and trf is not None:
        mism = (_r10(obj) or 0) >= 0.5 and (_r10(trf) or 0) < 0.1
        cov = (neg or {}).get("full_gallery_distractor_coverage_mean", {}).get("top100")
        H.append(("Objective↔full-gallery mismatch / weak negatives",
                  "Supported" if mism else "Insufficient",
                  f"objective R@10={_r10(obj)} vs train_full R@10={_r10(trf)}; top100 neg coverage={cov}"))
    if trf is not None and vaf is not None:
        memo = (_r10(trf) or 0) >= 0.5 and (_r10(vaf) or 0) < 0.1
        H.append(("Memorization / split domain shift",
                  "Supported" if memo else "Insufficient",
                  f"train_full R@10={_r10(trf)} vs val_full R@10={_r10(vaf)}"))
    if trf is not None:
        bug = (_r10(trf) or 0) < 0.1
        H.append(("Evaluation / cache / label / checkpoint bug",
                  "Insufficient",
                  "train_full low ⇒ see cache/checkpoint/labels audits" if bug
                  else "train_full not low ⇒ eval path is producing signal"))
    if ck is not None:
        clean = (not ck.get("missing_keys") and not ck.get("unexpected_keys")
                 and ck.get("eval_determinism_max_abs_diff", 1) < 1e-4
                 and not ck.get("any_dropout_training_in_eval"))
        H.append(("Checkpoint load + eval mode correct",
                  "Supported" if clean else "Refuted",
                  f"missing={ck.get('missing_keys')}, unexpected={ck.get('unexpected_keys')}, "
                  f"eval_det={ck.get('eval_determinism_max_abs_diff')}, "
                  f"dropout_in_eval={ck.get('any_dropout_training_in_eval')}"))
    if gate is not None:
        gc = (gate.get("train") or {}).get("potential_gate_collapse")
        H.append(("ScaleGate collapse", "Supported (flag)" if gc else "Refuted",
                  f"dominant_gate_frac={(gate.get('train') or {}).get('dominant_gate_frac')}"))
    if abl is not None:
        tl = abl.get("train") or {}
        lb = (tl.get("learned_beta") or {}).get("R@10")
        un = (tl.get("uniform_mean") or {}).get("R@10")
        mx = (tl.get("max_level") or {}).get("R@10")
        worse = lb is not None and ((un or 0) > (lb or 0) or (mx or 0) > (lb or 0))
        H.append(("Learned β underperforms uniform/max (gate suboptimal)",
                  "Supported" if worse else "Refuted",
                  f"learned R@10={lb}, uniform={un}, max={mx}"))

    lines = ["# Retrieval overfit diagnostic report", "",
             "| Hypothesis | Status | Evidence |", "|---|---|---|"]
    lines += [f"| {h} | {s} | {e} |" for h, s, e in H]
    lines += ["", "## Completion answers", "",
              f"1. High Recall on TRAIN over full gallery? "
              f"train_full R@10={_r10(trf)}, median_rank={(trf or {}).get('median_rank')}, "
              f"distR@500m={(trf or {}).get('distR@500m')}",
              f"2. Eval uses current checkpoint? "
              f"{'yes' if (cache and cache.get('eval_uses_current_checkpoint')) else 'see cache_audit.json'}",
              "3. Gap cause: read the Status column + decisive_summary.json['interpretation'] "
              "(kept multi-cause where evidence is insufficient)."]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(Path(out_dir) / "hypotheses.json",
               {"hypotheses": [{"hypothesis": h, "status": s, "evidence": e} for h, s, e in H]})
    return {"n_hypotheses": len(H)}
