"""Checkpoint round-trips, incompatible-mode error, resolved-config restore, §10 regression."""
import pytest
import torch

from siam_e2c_model.config import E2cModelConfig
from siam_e2c_model.model import build_e2c_model
from ._synth import build, grids, tokens, weights

FULL = (1000.0, 840.0, 710.0, 600.0, 500.0, 420.0, 350.0, 300.0, 250.0)


def _roundtrip(mode, kw):
    m1, _, _ = build("supervlad", d_out=16, pyramid_mode=mode, **kw)
    m2, _, _ = build("supervlad", d_out=16, pyramid_mode=mode, seed=123, **kw)  # different init
    m2.load_state_dict(m1.state_dict())
    m1.eval(); m2.eval()
    g = grids(m=3, g=6, d=64)
    V1, V2 = m1.build_V(g), m2.build_V(g)
    assert set(V1) == set(V2)
    for k in V1:
        assert torch.allclose(V1[k], V2[k], atol=1e-6)


def test_roundtrip_cell():
    _roundtrip("cell", {"scales": (4, 2, 1)})


def test_roundtrip_concentric():
    _roundtrip("concentric", {"concentric_sizes_m": FULL})


def test_incompatible_mode_raises_clear_error():
    cell, _, _ = build("supervlad", d_out=16, scales=(4, 2, 1))
    conc, _, _ = build("supervlad", d_out=16, pyramid_mode="concentric", concentric_sizes_m=FULL)
    with pytest.raises(RuntimeError, match="pyramid_mode"):
        cell.load_state_dict(conc.state_dict())        # scale_gate width + rho keys mismatch
    with pytest.raises(RuntimeError, match="pyramid_mode"):
        conc.load_state_dict(cell.state_dict())


def test_resolved_config_restores_architecture():
    m, _, _ = build("vlad", d_out=32, pyramid_mode="concentric", concentric_sizes_m=FULL)
    rc = m.resolved_config()
    assert rc["pyramid_mode"] == "concentric"
    assert rc["concentric_sizes_m"] == list(FULL)
    assert rc["tile_size_m"] == 1000.0
    assert rc["n_levels"] == 9
    assert rc["agg"] == "vlad"
    assert rc["level_values"] == list(FULL)
    # cell resolved config keeps scales_cells + cell-count level_values
    mc, _, _ = build("supervlad", scales=(4, 2, 1), d_out=16)
    rcc = mc.resolved_config()
    assert rcc["pyramid_mode"] == "cell"
    assert rcc["scales_cells"] == [4, 2, 1]
    assert rcc["level_values"] == [16, 4, 1]           # n² per scale


def test_concentric_order_normalized_recorded():
    # ascending input is explicitly canonicalised to descending and recorded
    m, _, _ = build("supervlad", d_out=16, pyramid_mode="concentric",
                    concentric_sizes_m=(250.0, 500.0, 1000.0))
    rc = m.resolved_config()
    assert rc["concentric_sizes_m"] == [1000.0, 500.0, 250.0]
    assert rc.get("concentric_order_normalized") is True
    assert rc["concentric_input_order"] == [250.0, 500.0, 1000.0]


def test_assign_weight_passed_with_centroids_from_file(tmp_path):
    # §10 regression: assign_weight given explicitly + centroids=None + assign_path set →
    # the vlad/residual arm must still load centroids from the blob (was skipped before).
    aw, cent = weights(k=6, d=64, seed=7)
    blob = {"assign_weight": aw, "centroids": cent}
    p = tmp_path / "k6.pt"
    torch.save(blob, p)
    cfg = E2cModelConfig(agg="vlad", k=6, d_token=64, scales_cells=(2, 1), d_group=8,
                         d_out=16, d_hidden=32, assign_path=str(p))
    m = build_e2c_model(cfg, assign_weight=aw, centroids=None)   # centroids come from the file
    m.eval()
    V = m.build_V(grids(m=2, g=6, d=64))
    assert tuple(V[1].shape) == (2, 1, 16)              # builds without "vlad needs centroids"
