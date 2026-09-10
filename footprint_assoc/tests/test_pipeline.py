import numpy as np

from footprint_assoc.config import FootprintConfig
from footprint_assoc import pyramid, kmz
from footprint_assoc.pipeline import estimate_frame, estimate_and_associate
from ._synth import make_pyramid, FakeExtractor, FakeCropSource


def test_estimate_picks_true_level():
    cfg = FootprintConfig()
    pyr = make_pyramid(true_idx=2)                 # true scale = 400 m
    est = estimate_frame(pyr, cfg)
    assert est.best_scale == pyr.scales()[2]
    assert abs(sum(est.soft.values()) - 1.0) < 1e-6


def test_estimate_deterministic():
    cfg = FootprintConfig()
    pyr = make_pyramid(true_idx=1)
    a = estimate_frame(pyr, cfg)
    b = estimate_frame(pyr, cfg)
    assert a.best_scale == b.best_scale and a.soft == b.soft


def test_rank_vote_mode_runs():
    cfg = FootprintConfig(fusion="rank_vote")
    est = estimate_frame(make_pyramid(true_idx=3), cfg)
    assert est.best_scale is not None


def test_build_pyramid_from_injected_extractor():
    cfg = FootprintConfig()
    scales = [100.0, 250.0, 1000.0]
    uav = np.full((8, 8, 3), 250, dtype=np.uint8)   # same "content" as the 250 m crop
    pyr = pyramid.build("f1", 48.5, 37.8, uav, scales, FakeCropSource(), FakeExtractor())
    assert pyr.scales() == scales
    est = estimate_frame(pyr, cfg)
    assert est.best_scale == 250.0                  # matches the crop with identical features


def test_estimate_and_associate_and_kmz(tmp_path):
    cfg = FootprintConfig()
    pyr = make_pyramid(true_idx=2)                 # 400 m
    est, pairs = estimate_and_associate(pyr, [("t0", 48.5, 37.8, 400.0)], cfg)
    assert any(p.tile_id == "t0" for p in pairs)
    tiles_geom = {"t0": (48.5, 37.8, 400.0)}
    p = kmz.write_kmz(tmp_path / "v.kmz", [est], tiles_geom=tiles_geom,
                      assoc={est.frame_id: pairs})
    import zipfile
    assert zipfile.is_zipfile(p)
    with zipfile.ZipFile(p) as z:
        doc = z.read("doc.kml").decode()
    assert "f0" in doc and "<Polygon>" in doc
