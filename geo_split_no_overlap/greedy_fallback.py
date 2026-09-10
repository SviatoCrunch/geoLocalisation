"""Deterministic greedy fallback for when no exact solver (OR-Tools) is available.

Same two-phase contract as the CP-SAT backend but heuristic (``optimality_proven``
is always False). It is deterministic for identical inputs (fixed orderings, string
tie-breaks — never Python ``hash()``). Assignment balances by POINT count; deletion
excludes WHOLE components, smallest-first, to minimise removed points.
"""
from __future__ import annotations

from .schemas import TRAIN, VAL, TEST


def _targets(ratios, total_kept):
    return {0: ratios[TRAIN] * total_kept, 1: ratios[VAL] * total_kept, 2: ratios[TEST] * total_kept}


def _assign(kept, sizes, names, ratios):
    """Greedy: biggest components first, each to the currently most-under-target split."""
    total_kept = sum(sizes[c] for c in kept)
    tgt = _targets(ratios, total_kept)
    got = {0: 0, 1: 0, 2: 0}
    assign = {}
    for c in sorted(kept, key=lambda c: (-sizes[c], names[c])):
        # most-under-target split; deterministic tie-break by split index
        s = max((tgt[s] - got[s], -s) for s in (0, 1, 2))[1]
        s = -s
        assign[c] = s
        got[s] += sizes[c]
    return assign, got, total_kept


def _within_tol(got, total_kept, ratios, tol):
    if total_kept <= 0:
        return False
    r = {0: ratios[TRAIN], 1: ratios[VAL], 2: ratios[TEST]}
    for s in (0, 1, 2):
        if got[s] < 1:
            return False
        if abs(got[s] / total_kept - r[s]) > tol + 1e-12:
            return False
    return True


def _max_dev(got, total_kept, ratios):
    if total_kept <= 0:
        return float("inf")
    r = {0: ratios[TRAIN], 1: ratios[VAL], 2: ratios[TEST]}
    return max(abs(got[s] / total_kept - r[s]) for s in (0, 1, 2))


def run_greedy(sizes, names, ratios, tol, seed, allow_deletion):
    n = len(sizes)
    kept = set(range(n))

    assign, got, total_kept = _assign(kept, sizes, names, ratios)
    if _within_tol(got, total_kept, ratios, tol):
        return _result(assign, kept, n, sizes)

    if not allow_deletion:
        return {"status": "infeasible"}

    # Phase B: exclude whole components, choosing at each step the deletion that most
    # reduces imbalance; ties -> smallest component, then name (minimise removed points).
    removed_points = 0
    while len(kept) > 3:
        best = None
        for c in sorted(kept, key=lambda c: (sizes[c], names[c])):
            trial = kept - {c}
            _, g2, tk2 = _assign(trial, sizes, names, ratios)
            if tk2 <= 0 or any(g2[s] < 1 for s in (0, 1, 2)):
                continue
            md = _max_dev(g2, tk2, ratios)
            key = (md, sizes[c], names[c])
            if best is None or key < best[0]:
                best = (key, c, g2, tk2)
        if best is None:
            break
        _, c, got, total_kept = best
        kept.discard(c)
        removed_points += sizes[c]
        if _within_tol(got, total_kept, ratios, tol):
            assign, _, _ = _assign(kept, sizes, names, ratios)
            res = _result(assign, kept, n, sizes)
            res["stage_objectives"] = {"removed_points": removed_points}
            return res

    return {"status": "infeasible"}


def _result(assign, kept, n, sizes):
    assign_idx = {c: assign[c] for c in range(n) if c in kept}
    for c in range(n):
        if c not in kept:
            assign_idx[c] = 3                        # excluded
    return {"status": "ok", "assign_idx": assign_idx, "optimality_proven": False,
            "stage_objectives": {}}
