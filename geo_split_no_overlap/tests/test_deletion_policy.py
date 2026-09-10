import pytest

from geo_split_no_overlap import optimizer
from geo_split_no_overlap.optimizer import have_ortools
from geo_split_no_overlap.schemas import TRAIN, VAL, TEST, EXCLUDED
from ._synth import comps

SOLVERS = ["heuristic"] + (["cpsat"] if have_ortools() else [])
EQUAL = {TRAIN: 1 / 3, VAL: 1 / 3, TEST: 1 / 3}


@pytest.mark.parametrize("solver", SOLVERS)
def test_deletion_only_when_needed_and_smallest(solver):
    # 3x size-10 fit perfectly if the size-1 is removed; keeping it breaks tol=0.02.
    components = comps([10, 10, 10, 1])
    res = optimizer.solve(components, EQUAL, tolerance=0.02, seed=0, solver=solver)
    assert res.status == "ok"
    assert res.phase == "B_deletion"
    assert res.deletion_used is True
    excluded = [c for c in components if res.assignment[c.component_id] == EXCLUDED]
    assert len(excluded) == 1 and excluded[0].size == 1        # smallest removed


@pytest.mark.parametrize("solver", SOLVERS)
def test_no_deletion_when_tolerance_already_met(solver):
    components = comps([10, 10, 10, 1])
    res = optimizer.solve(components, EQUAL, tolerance=0.10, seed=0, solver=solver)
    assert res.status == "ok"
    assert res.phase == "A_no_deletion"
    assert res.deletion_used is False


@pytest.mark.parametrize("solver", SOLVERS)
def test_large_component_makes_ratios_infeasible(solver):
    components = comps([100, 1, 1])                            # 100 can't be split -> infeasible
    res = optimizer.solve(components, {TRAIN: 0.7, VAL: 0.15, TEST: 0.15},
                          tolerance=0.05, seed=0, solver=solver)
    assert res.status == "infeasible"


@pytest.mark.skipif(not have_ortools(), reason="OR-Tools not installed")
def test_cpsat_minimises_removed_points_over_component_count():
    # Removing one size-2 (2 pts) beats removing two size-1 (2 pts too) -> tie broken
    # by max-removed-component then sumsq; but total removed points is the first key.
    # Here removing the single size-3 (3 pts) is worse than removing size-1+size-1? craft:
    components = comps([10, 10, 10, 1, 1])                     # remove BOTH size-1 => 30 balanced
    res = optimizer.solve(components, EQUAL, tolerance=0.02, seed=0, solver="cpsat")
    assert res.status == "ok"
    removed_pts = sum(c.size for c in components
                      if res.assignment[c.component_id] == EXCLUDED)
    assert removed_pts == 2                                    # minimal removed points
