import pytest

from geo_split_no_overlap import optimizer
from geo_split_no_overlap.optimizer import have_ortools
from geo_split_no_overlap.schemas import TRAIN, VAL, TEST, EXCLUDED
from ._synth import comps

SOLVERS = ["heuristic"] + (["cpsat"] if have_ortools() else [])
RATIOS = {TRAIN: 0.7, VAL: 0.15, TEST: 0.15}


def _counts(components, res):
    c = {TRAIN: 0, VAL: 0, TEST: 0, EXCLUDED: 0}
    for comp in components:
        c[res.assignment[comp.component_id]] += comp.size
    return c


@pytest.mark.parametrize("solver", SOLVERS)
def test_distributes_without_deletion(solver):
    components = comps([70, 15, 15, 5, 5, 5, 10, 10, 10, 5])   # total 150
    res = optimizer.solve(components, RATIOS, tolerance=0.05, seed=0, solver=solver)
    assert res.status == "ok"
    assert res.phase == "A_no_deletion"
    assert res.deletion_used is False
    c = _counts(components, res)
    kept = c[TRAIN] + c[VAL] + c[TEST]
    assert c[EXCLUDED] == 0 and kept == 150
    for s in (TRAIN, VAL, TEST):
        assert abs(c[s] / kept - RATIOS[s]) <= 0.05 + 1e-9


@pytest.mark.parametrize("solver", SOLVERS)
def test_deterministic_same_seed(solver):
    components = comps([70, 15, 15, 5, 5, 5, 10, 10, 10, 5])
    r1 = optimizer.solve(components, RATIOS, 0.05, seed=7, solver=solver)
    r2 = optimizer.solve(components, RATIOS, 0.05, seed=7, solver=solver)
    assert r1.assignment == r2.assignment


def test_fewer_than_three_components_infeasible():
    res = optimizer.solve(comps([5, 5]), RATIOS, 0.05, solver="heuristic")
    assert res.status == "infeasible"


@pytest.mark.skipif(not have_ortools(), reason="OR-Tools not installed")
def test_cpsat_reports_optimality():
    components = comps([70, 15, 15, 10, 10, 10, 10, 10])
    res = optimizer.solve(components, RATIOS, 0.05, seed=0, solver="cpsat")
    assert res.backend == "ortools_cpsat"
    assert res.optimality_proven is True
