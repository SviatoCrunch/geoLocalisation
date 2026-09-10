from footprint_assoc.config import FootprintConfig
from footprint_assoc.fusion import available_fusion, create_fusion
from footprint_assoc.schemas import ACCEPT, SOFT, REFUSE, LevelScore
from ._synth import per_method_peaked

SCALES = [100.0, 200.0, 400.0, 800.0, 1000.0]


def test_registry():
    assert "cascade" in available_fusion() and "rank_vote" in available_fusion()


def test_cascade_accepts_on_agreement_and_margin():
    cfg = FootprintConfig(tau=0.05, accept_margin=0.02)
    pm = per_method_peaked(SCALES, g_peak=2, p_peak=2, g_val=0.95, p_val=0.9)
    est = create_fusion("cascade").fuse("f", 48.5, 37.8, SCALES, pm, cfg)
    assert est.status == ACCEPT
    assert est.best_scale == 400.0
    assert abs(sum(est.soft.values()) - 1.0) < 1e-6


def test_cascade_soft_on_disagreement():
    cfg = FootprintConfig(agreement_levels=1)
    pm = per_method_peaked(SCALES, g_peak=0, p_peak=4)   # methods point far apart
    est = create_fusion("cascade").fuse("f", 48.5, 37.8, SCALES, pm, cfg)
    assert est.status == SOFT


def test_cascade_refuses_on_zero_evidence():
    cfg = FootprintConfig()
    pm = {"global_vlad": [], "patch_overlap": []}
    est = create_fusion("cascade").fuse("f", 48.5, 37.8, SCALES, pm, cfg)
    assert est.status == REFUSE and est.best_scale is None and est.confidence == 0.0


def test_rank_vote_scale_invariant_pick():
    cfg = FootprintConfig(fusion="rank_vote", accept_margin=0.0)
    # global has huge magnitudes, patch tiny — but both RANK idx1 first -> RRF picks it
    pm = {"global_vlad": [LevelScore(s, 1000.0 if i == 1 else 1.0) for i, s in enumerate(SCALES)],
          "patch_overlap": [LevelScore(s, 0.02 if i == 1 else 0.001) for i, s in enumerate(SCALES)]}
    est = create_fusion("rank_vote").fuse("f", 48.5, 37.8, SCALES, pm, cfg)
    assert est.best_scale == 200.0
