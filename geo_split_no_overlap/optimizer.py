"""Assign indivisible components to train/val/test, deleting whole components only.

This is a component-level weighted assignment, NOT a vertex separator: each component
is atomic (goes whole to one split or is excluded whole), and balance is by POINT
count. We deliberately do not reuse ``siam_model_stage2.graph_split`` for the core —
that solver is a *vertex separator* whose constraint is satisfied by removing
individual boundary points, which this task forbids. We do mirror its proven
conventions: an exact backend when available (OR-Tools CP-SAT) with a deterministic
greedy fallback, a staged lexicographic objective, and a SHA-256 canonical tie-break.

Two phases (spec §4–5):
  A  no deletion — assign all components; accept iff every split is within tolerance.
  B  deletion allowed (only if A failed) — lexicographically:
       1. minimise removed points
       2. minimise the largest removed component
       3. minimise the sum of squared removed-component sizes (prefers small ones)
       4. minimise the worst split imbalance (nicety)
       5. SHA-256 canonical tie-break -> a unique, reproducible assignment
Balance (within ``ratio_tolerance``) and non-empty kept splits are HARD constraints.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .schemas import TRAIN, VAL, TEST, EXCLUDED, KEPT_SPLITS

SCALE = 10_000
_SPLIT_IDX = {TRAIN: 0, VAL: 1, TEST: 2}
_IDX_SPLIT = {0: TRAIN, 1: VAL, 2: TEST, 3: EXCLUDED}


@dataclass
class OptimizerResult:
    assignment: dict                 # component_id -> split/excluded
    backend: str
    status: str                      # "ok" | "infeasible"
    optimality_proven: bool
    phase: str                       # "A_no_deletion" | "B_deletion"
    deletion_used: bool
    stage_objectives: dict = field(default_factory=dict)
    message: str = ""


def have_ortools() -> bool:
    try:
        from ortools.sat.python import cp_model  # noqa: F401
        return True
    except Exception:
        return False


def scaled_fracs(ratios: dict) -> dict:
    a_train = int(round(ratios[TRAIN] * SCALE))
    a_val = int(round(ratios[VAL] * SCALE))
    a_test = SCALE - a_train - a_val            # test absorbs rounding
    return {TRAIN: a_train, VAL: a_val, TEST: a_test}


def canonical_weight(seed: int, name: str, idx: int) -> int:
    """Stable ~40-bit weight for (component, assignment-index) from SHA-256.

    Never Python ``hash()`` (salted). Depends on both the component identity and the
    chosen slot, so minimising it fixes the whole assignment deterministically.
    """
    h = hashlib.sha256(f"{int(seed)}|{name}|{int(idx)}".encode()).hexdigest()
    return int(h[:10], 16)


def solve(components, ratios: dict, tolerance: float, seed: int = 0,
          solver: str = "auto", time_limit_s: float = 60.0, workers: int = 8) -> OptimizerResult:
    """Two-phase solve. ``components`` is a list of objects with ``.size`` and a stable
    name (``.point_ids[0]`` is used as the canonical name)."""
    sizes = [int(c.size) for c in components]
    names = [c.point_ids[0] for c in components]
    ids = [c.component_id for c in components]

    if len(components) < 3:
        return OptimizerResult({}, "none", "infeasible", False, "A_no_deletion", False,
                               message=f"only {len(components)} component(s); cannot fill 3 non-empty splits")

    backend = _pick_backend(solver)

    # Phase A: no deletion.
    resA = _run(backend, sizes, names, ratios, tolerance, seed, time_limit_s, workers,
                allow_deletion=False)
    if resA is not None and resA["status"] == "ok":
        return _finalize(ids, resA, backend, phase="A_no_deletion", deletion_used=False)

    # Phase B: deletion allowed.
    resB = _run(backend, sizes, names, ratios, tolerance, seed, time_limit_s, workers,
                allow_deletion=True)
    if resB is None or resB["status"] != "ok":
        return OptimizerResult({}, backend, "infeasible", False, "B_deletion", True,
                               message="no valid non-empty split exists within tolerance, "
                                       "even after excluding whole components")
    return _finalize(ids, resB, backend, phase="B_deletion",
                     deletion_used=any(v == 3 for v in resB["assign_idx"].values()))


def _pick_backend(solver: str) -> str:
    if solver in ("auto", "exact", "cpsat"):
        if have_ortools():
            return "ortools_cpsat"
        if solver in ("exact", "cpsat"):
            raise RuntimeError("solver=%r requested but OR-Tools CP-SAT is not importable" % solver)
        return "heuristic"                       # auto -> greedy
    if solver == "heuristic":
        return "heuristic"
    if solver == "milp":
        raise RuntimeError("solver='milp' (scipy) backend is not implemented; "
                           "use 'cpsat'/'exact' (OR-Tools) or 'heuristic'")
    raise ValueError(f"unknown solver {solver!r}")


def _run(backend, sizes, names, ratios, tol, seed, tl, workers, allow_deletion):
    if backend == "ortools_cpsat":
        return _run_cpsat(sizes, names, ratios, tol, seed, tl, workers, allow_deletion)
    from .greedy_fallback import run_greedy
    return run_greedy(sizes, names, ratios, tol, seed, allow_deletion)


def _finalize(ids, res, backend, phase, deletion_used) -> OptimizerResult:
    assignment = {ids[c]: _IDX_SPLIT[idx] for c, idx in res["assign_idx"].items()}
    return OptimizerResult(assignment=assignment, backend=backend, status="ok",
                           optimality_proven=res.get("optimality_proven", False),
                           phase=phase, deletion_used=deletion_used,
                           stage_objectives=res.get("stage_objectives", {}),
                           message=res.get("message", ""))


# ── OR-Tools CP-SAT backend ──────────────────────────────────────────────────────
def _run_cpsat(sizes, names, ratios, tol, seed, time_limit_s, workers, allow_deletion):
    from ortools.sat.python import cp_model

    n = len(sizes)
    total = sum(sizes)
    a = scaled_fracs(ratios)
    TOL = int(round(tol * SCALE))

    m = cp_model.CpModel()
    x = {(c, s): m.NewBoolVar(f"x_{c}_{s}") for c in range(n) for s in range(3)}
    d = {c: m.NewBoolVar(f"d_{c}") for c in range(n)}

    for c in range(n):
        m.Add(sum(x[c, s] for s in range(3)) + d[c] == 1)
        if not allow_deletion:
            m.Add(d[c] == 0)

    kept = {s: sum(sizes[c] * x[c, s] for c in range(n)) for s in range(3)}
    K = m.NewIntVar(1, total, "K")
    m.Add(K == sum(kept[s] for s in range(3)))

    a_by_idx = {0: a[TRAIN], 1: a[VAL], 2: a[TEST]}
    for s in range(3):
        m.Add(kept[s] >= 1)                                      # non-empty kept split
        # |SCALE*kept - a_s*K| <= TOL*K   (linear: TOL, a_s, SCALE are constants)
        m.Add(SCALE * kept[s] - a_by_idx[s] * K <= TOL * K)
        m.Add(a_by_idx[s] * K - SCALE * kept[s] <= TOL * K)

    # deviation vars for the balance-quality objective
    dev = {s: m.NewIntVar(0, SCALE * total, f"dev_{s}") for s in range(3)}
    for s in range(3):
        m.Add(dev[s] >= SCALE * kept[s] - a_by_idx[s] * K)
        m.Add(dev[s] >= a_by_idx[s] * K - SCALE * kept[s])
    max_dev = m.NewIntVar(0, SCALE * total, "max_dev")
    for s in range(3):
        m.Add(max_dev >= dev[s])

    # lexicographic objective stages (each an int linear expr)
    stages = []
    if allow_deletion:
        removed = sum(sizes[c] * d[c] for c in range(n))
        max_rem = m.NewIntVar(0, total, "max_rem")
        for c in range(n):
            m.Add(max_rem >= sizes[c] * d[c])
        sumsq = sum((sizes[c] * sizes[c]) * d[c] for c in range(n))
        stages += [("removed_points", removed),
                   ("max_removed_component", max_rem),
                   ("sumsq_removed", sumsq)]
    stages.append(("max_imbalance", max_dev))
    canonical = sum(canonical_weight(seed, names[c], s) * x[c, s]
                    for c in range(n) for s in range(3))
    canonical += sum(canonical_weight(seed, names[c], 3) * d[c] for c in range(n))
    stages.append(("canonical", canonical))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(1.0, time_limit_s / max(1, len(stages)))
    solver.parameters.num_search_workers = max(1, int(workers))
    solver.parameters.random_seed = int(seed)

    stage_objectives = {}
    optimality_proven = True
    for name, expr in stages:
        m.Minimize(expr)
        st = solver.Solve(m)
        if st == cp_model.INFEASIBLE:
            return None
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        if st != cp_model.OPTIMAL:
            optimality_proven = False
        val = int(round(solver.ObjectiveValue()))
        stage_objectives[name] = val
        m.Add(expr == val)                                       # lock this stage's optimum
        # next m.Minimize(...) overwrites the objective, so no explicit clear is needed

    assign_idx = {}
    for c in range(n):
        if allow_deletion and solver.Value(d[c]) == 1:
            assign_idx[c] = 3
        else:
            assign_idx[c] = next(s for s in range(3) if solver.Value(x[c, s]) == 1)

    return {"status": "ok", "assign_idx": assign_idx,
            "optimality_proven": optimality_proven, "stage_objectives": stage_objectives}
