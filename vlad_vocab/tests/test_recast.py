"""Recast blob is the e2c dictionary: assign_weight (supervlad) + centroids (vlad).

End-to-end: the saved k<K>.pt must load through the SAME paths build_e2c_model uses —
assign_io.load_assign_weight (supervlad) and VladAggregation via assign_path (vlad) — so a
fresh model builds from the file alone (also covers the assign_path centroids path).
"""
import torch

from vlad_vocab.recast import recast_centroids


def test_recast_shapes_and_supervlad_relation():
    torch.manual_seed(0)
    C = torch.randn(6, 32)
    X = torch.randn(400, 32)
    blob = recast_centroids(C, X)
    assert tuple(blob["assign_weight"].shape) == (6, 32)
    assert tuple(blob["centroids"].shape) == (6, 32)
    # assign_weight == γ · L2(centroids); centroids stored L2-normalised
    import torch.nn.functional as F
    Cn = F.normalize(C, dim=1)
    assert torch.allclose(blob["centroids"], Cn, atol=1e-5)
    assert torch.allclose(blob["assign_weight"], float(blob["alpha"]) * Cn, atol=1e-4)
    assert blob["alpha"] > 0 and torch.isfinite(blob["assign_weight"]).all()


def test_saved_blob_loads_via_assign_io(tmp_path):
    from siam_e2c_model.vendored.assign_io import load_assign_weight
    C = torch.randn(6, 32)
    blob = recast_centroids(C, torch.randn(300, 32))
    p = tmp_path / "k6.pt"
    torch.save(blob, p)
    w = load_assign_weight(p, k=6, d=32)
    assert tuple(w.shape) == (6, 32)


def test_build_e2c_model_from_assign_path_both_arms(tmp_path):
    # the file alone must build supervlad AND vlad/residual (assign_weight + centroids).
    from siam_e2c_model.config import E2cModelConfig
    from siam_e2c_model.model import build_e2c_model
    from vlad_vocab.tests._synth_ok import grids
    blob = recast_centroids(torch.randn(6, 32), torch.randn(300, 32))
    p = tmp_path / "k6.pt"
    torch.save(blob, p)
    for agg in ("supervlad", "vlad", "residual"):
        cfg = E2cModelConfig(agg=agg, k=6, d_token=32, scales_cells=(2, 1),
                             d_group=8, d_out=16, d_hidden=32, assign_path=str(p))
        m = build_e2c_model(cfg)             # NO assign_weight/centroids passed -> loaded from file
        m.eval()
        V = m.build_V(grids(m=2, g=4, d=32))
        assert set(V) == {2, 1}
        assert tuple(V[1].shape) == (2, 1, 16)
