from footprint_assoc.config import FootprintConfig
from footprint_assoc.association import associate
from footprint_assoc.schemas import (FrameEstimate, ACCEPT, REFUSE, STRONG, NEGATIVE,
                                     UNDETERMINED)

LAT, LON = 48.5, 37.8
SCALES = [200.0, 500.0, 1000.0]


def _estimate(status=ACCEPT, best=500.0):
    soft = {s: (1.0 if s == best else 0.0) for s in SCALES}
    return FrameEstimate("f", LAT, LON, SCALES, {}, soft,
                         best if status != REFUSE else None,
                         1.0 if status == ACCEPT else 0.0, status)


def test_concentric_tile_is_strong():
    est = _estimate(best=500.0)
    # tile concentric with the frame, same 500 m size -> full footprint overlap
    res = associate(est, [("t0", LAT, LON, 500.0)], FootprintConfig())
    assert res[0].status == STRONG and res[0].cq > 0.99


def test_far_tile_is_negative():
    est = _estimate(best=500.0)
    res = associate(est, [("far", 49.5, 40.0, 500.0)], FootprintConfig())
    assert res[0].status == NEGATIVE and res[0].iou == 0.0


def test_refused_frame_yields_no_hard_labels():
    est = _estimate(status=REFUSE)
    res = associate(est, [("t0", LAT, LON, 500.0), ("far", 49.5, 40.0, 500.0)],
                    FootprintConfig())
    by = {r.tile_id: r for r in res}
    assert by["t0"].status == UNDETERMINED and by["t0"].weight == 0.0
    assert by["far"].status == NEGATIVE


def test_partial_overlap_graded_weight():
    from footprint_assoc.schemas import PARTIAL
    est = _estimate(best=1000.0)
    # small offset: 500 m tile sits inside the 1000 m footprint -> partial (cq~0.25)
    res = associate(est, [("p", LAT + 0.001, LON, 500.0)], FootprintConfig())
    r = res[0]
    assert 0.0 < r.iou < 1.0 and r.status == PARTIAL and r.weight > 0.0
