from geo_split_no_overlap.spatial_conflicts import (
    rect_intersection_area, overlap_true_m2, build_tile_clusters)
from ._synth import tile, make_gallery


def test_rect_intersection_basic():
    assert rect_intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25.0
    assert rect_intersection_area((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_touching_boundary_is_zero_area():
    assert rect_intersection_area((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_overlap_true_m2_touching_is_zero():
    a = tile("a", 0.0, 0.0, size=1000.0)
    b = tile("b", 1000.0, 0.0, size=1000.0)
    assert overlap_true_m2(a, b, 1000.0) == 0.0


def test_overlapping_tiles_cluster_together():
    g = make_gallery([tile("a", 0.0, 0.0), tile("b", 250.0, 0.0)])
    cluster_of, n, edges = build_tile_clusters(g, ["a", "b"], area_epsilon_m2=1.0,
                                               tile_size_fallback=1000.0)
    assert n == 1 and edges == 1
    assert cluster_of["a"] == cluster_of["b"]


def test_touching_tiles_are_separate_clusters():
    g = make_gallery([tile("a", 0.0, 0.0), tile("b", 1000.0, 0.0)])
    cluster_of, n, edges = build_tile_clusters(g, ["a", "b"], area_epsilon_m2=1.0,
                                               tile_size_fallback=1000.0)
    assert n == 2 and edges == 0
    assert cluster_of["a"] != cluster_of["b"]


def test_area_epsilon_threshold():
    g = make_gallery([tile("a", 0.0, 0.0), tile("b", 999.0, 0.0)])
    small = build_tile_clusters(g, ["a", "b"], area_epsilon_m2=1e12,
                                tile_size_fallback=1000.0)[1]
    big = build_tile_clusters(g, ["a", "b"], area_epsilon_m2=1.0,
                              tile_size_fallback=1000.0)[1]
    assert small == 2 and big == 1
