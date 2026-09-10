import numpy as np

from footprint_assoc.config import FootprintConfig
from footprint_assoc.estimators import (available_estimators, create_estimator)
from ._synth import make_pyramid


def test_registry():
    names = available_estimators()
    assert "global_vlad" in names and "patch_overlap" in names
    assert create_estimator("global_vlad").name == "global_vlad"


def test_global_vlad_peaks_at_true_level():
    cfg = FootprintConfig()
    pyr = make_pyramid(true_idx=2)
    est = create_estimator("global_vlad")
    scores = [est.score_level(pyr.query, lv, cfg).score for lv in pyr.levels]
    assert int(np.argmax(scores)) == 2
    assert scores[2] > 0.99                     # cosine ~1 at the matching level


def test_patch_overlap_peaks_at_true_level():
    cfg = FootprintConfig()
    pyr = make_pyramid(true_idx=3)
    est = create_estimator("patch_overlap")
    out = [est.score_level(pyr.query, lv, cfg) for lv in pyr.levels]
    scores = [o.score for o in out]
    assert int(np.argmax(scores)) == 3
    assert out[3].aux["cq"] > 0.8               # nearly all query patches supported at true level


def test_patch_overlap_empty_grid_zero():
    cfg = FootprintConfig()
    pyr = make_pyramid(true_idx=0)
    est = create_estimator("patch_overlap")
    from footprint_assoc.schemas import LevelFeatures
    empty = LevelFeatures(scale_m=100.0, global_vec=pyr.levels[0].global_vec,
                          patch_grid=np.zeros((0, 0, 16)))
    assert est.score_level(pyr.query, empty, cfg).score == 0.0
