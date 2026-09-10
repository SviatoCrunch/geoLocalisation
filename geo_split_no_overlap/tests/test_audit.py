from geo_split_no_overlap.audit import audit
from geo_split_no_overlap.schemas import TRAIN, VAL, TEST
from ._synth import cfg, tile, make_gallery, make_mps


def test_valid_split_passes_audit():
    g = make_gallery([tile("t0", 0.0, 0.0), tile("t1", 1_000_000.0, 0.0),
                      tile("t2", 2_000_000.0, 0.0)])
    mps = make_mps({"a": ["t0"], "b": ["t0"], "c": ["t1"], "d": ["t2"]}, g)
    point_split = {"a": TRAIN, "b": TRAIN, "c": VAL, "d": TEST}
    rep = audit(cfg(tol=1.0), g, mps, point_split)
    assert rep.status == "valid"
    assert rep.cross_split_conflicts == 0
    assert all(v["ok"] for v in rep.checks.values())


def test_shared_tile_across_splits_is_invalid():
    g = make_gallery([tile("t0", 0.0, 0.0), tile("t2", 1_000_000.0, 0.0),
                      tile("t3", 2_000_000.0, 0.0)])
    mps = make_mps({"a": ["t0"], "b": ["t0"], "c": ["t2"], "d": ["t3"]}, g)
    point_split = {"a": TRAIN, "b": VAL, "c": VAL, "d": TEST}   # a,b share t0
    rep = audit(cfg(tol=1.0), g, mps, point_split)
    assert rep.status == "invalid"
    assert not rep.checks["tile_id_disjoint"]["ok"]
    assert not rep.checks["components_atomic"]["ok"]


def test_overlapping_tiles_across_splits_is_invalid():
    g = make_gallery([tile("t0", 0.0, 0.0), tile("t1", 250.0, 0.0),
                      tile("t2", 1_000_000.0, 0.0), tile("t3", 2_000_000.0, 0.0)])
    mps = make_mps({"a": ["t0"], "b": ["t1"], "c": ["t2"], "d": ["t3"]}, g)
    point_split = {"a": TRAIN, "b": VAL, "c": VAL, "d": TEST}   # t0,t1 overlap
    rep = audit(cfg(tol=1.0), g, mps, point_split)
    assert rep.status == "invalid"
    assert not rep.checks["geo_disjoint"]["ok"]
    assert rep.cross_split_conflicts >= 1
