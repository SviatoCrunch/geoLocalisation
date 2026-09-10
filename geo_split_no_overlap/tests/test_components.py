from geo_split_no_overlap.components import build_components
from ._synth import tile, make_gallery, make_mps

EPS, FALLBACK = 1.0, 1000.0


def _build(point_to_tiles, tiles):
    g = make_gallery(tiles)
    mps = make_mps(point_to_tiles, g)
    return build_components(g, mps, EPS, FALLBACK)


def test_same_tile_shared_is_one_component():
    comps = _build({"a": ["t0"], "b": ["t0"]}, [tile("t0", 0.0, 0.0)])
    assert len(comps) == 1
    assert set(comps[0].point_ids) == {"a", "b"}


def test_different_overlapping_tiles_is_one_component():
    comps = _build({"a": ["t0"], "b": ["t1"]},
                   [tile("t0", 0.0, 0.0), tile("t1", 250.0, 0.0)])
    assert len(comps) == 1


def test_touching_only_tiles_are_two_components():
    comps = _build({"a": ["t0"], "b": ["t1"]},
                   [tile("t0", 0.0, 0.0), tile("t1", 1000.0, 0.0)])
    assert len(comps) == 2


def test_transitive_chain_is_one_component():
    comps = _build({"a": ["t0"], "b": ["t1"], "c": ["t2"]},
                   [tile("t0", 0.0, 0.0), tile("t1", 900.0, 0.0), tile("t2", 1800.0, 0.0)])
    assert len(comps) == 1
    assert set(comps[0].point_ids) == {"a", "b", "c"}


def test_independent_components():
    comps = _build({"a": ["t0"], "b": ["t0"], "c": ["t1"]},
                   [tile("t0", 0.0, 0.0), tile("t1", 1_000_000.0, 0.0)])
    assert len(comps) == 2
    assert {c.size for c in comps} == {2, 1}


def test_component_ids_are_stable_and_sorted():
    comps = _build({"z": ["t1"], "a": ["t0"]},
                   [tile("t0", 0.0, 0.0), tile("t1", 1_000_000.0, 0.0)])
    assert comps[0].point_ids[0] == "a"
