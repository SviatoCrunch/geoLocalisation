"""8.9 mocked end-to-end forward/backward through the NATIVE source + cache, plus config guards.

Exercises the real runtime path: NativeCellCache → NativeHierarchicalSource.build_V →
model.build_V_from_cell_sums → score → backward. No GPU, no COG, no DINO (synthetic S cache).

Run: cd geoLocalisation && python -m pytest native_map_pyramid/tests/test_smoke_e2e.py -q
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from siam_e2c_model.config import E2cModelConfig
from siam_e2c_model.vendored.stage2_core import build_stage2_query_conditioned_model
from native_map_pyramid import cache as C
from native_map_pyramid.source import NativeHierarchicalSource


def _model(K=4, Dv=16, d_out=8, seed=0):
    torch.manual_seed(seed)
    assign = F.normalize(torch.randn(K, Dv), dim=1)
    return build_stage2_query_conditioned_model(
        d_token=Dv, n_groups=K, n_ghost=0, group_projection_dim=4, scales_cells=(8, 4, 2, 1),
        d_out=d_out, head_hidden=16, dropout=0.0, assign_weight=assign, freeze_assignment=True)


def _write_synth_cache(tmp_path, ids, K=4, Dv=16, seed=1):
    rng = np.random.RandomState(seed)
    S = rng.randn(len(ids), C.N_CELLS_TOTAL, K, Dv).astype(np.float32)
    ident = C.build_fingerprint(backbone="dinov2_vitg14", dino_layer=None, dino_facet="value",
                                output_px=840, tile_size_m=1000.0, patch_size=14,
                                projection="random_gaussian_jl", projection_seed=0,
                                projection_in_dim=32, projection_out_dim=Dv, n_groups=K,
                                d_value=Dv, vlad_dict_id="unit")
    path = tmp_path / "S.h5"
    C.write_cache(path, S, ids, ident)
    return path


def test_native_forward_backward_smoke(tmp_path):
    K, Dv, d_out, B = 4, 16, 8, 3
    ids = [f"c:{i}" for i in range(6)]
    model = _model(K=K, Dv=Dv, d_out=d_out)
    model.train()
    cache = C.NativeCellCache(_write_synth_cache(tmp_path, ids, K=K, Dv=Dv))
    src = NativeHierarchicalSource(cache, "cpu")

    V = src.build_V(model, ids)                          # {n:(M,n²,d_out)}
    assert V[8].shape == (6, 64, d_out) and V[4].shape == (6, 16, d_out)
    assert V[2].shape == (6, 4, d_out) and V[1].shape == (6, 1, d_out)
    for n, Vn in V.items():                               # L2-normalized cell descriptors
        assert torch.allclose(Vn.norm(dim=-1), torch.ones(6, n * n), atol=1e-4)

    Q = F.normalize(torch.randn(B, d_out), dim=-1)
    scores = model.score_queries_against_tiles(Q, V)     # (B, M)
    assert scores.shape == (B, 6)
    loss = F.cross_entropy(scores, torch.randint(0, 6, (B,)))
    assert torch.isfinite(loss)

    loss.backward()
    # gradients reach the trainable tail …
    for p in (model.group_proj.weight, model.map_head.net[0].weight,
              model.scale_gate.net[0].weight, model.rho[str(8)]):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    # … but NOT the frozen assignment (also unused on the native path)
    assert model.agg.assign.weight.grad is None
    cache.close()


def test_native_matches_two_calls_are_deterministic(tmp_path):
    ids = [f"c:{i}" for i in range(4)]
    model = _model()
    model.eval()
    cache = C.NativeCellCache(_write_synth_cache(tmp_path, ids))
    src = NativeHierarchicalSource(cache, "cpu")
    with torch.no_grad():
        V1 = src.build_V(model, ids)
        V2 = src.build_V(model, ids)
    for n in (8, 4, 2, 1):
        assert torch.allclose(V1[n], V2[n])
    cache.close()


# ── config guards: default preserves legacy; bad value rejected ─────────────────────────
def test_config_default_is_legacy():
    assert E2cModelConfig().map_pyramid_source == "legacy_token_partition"
    E2cModelConfig().validate()                          # default validates


def test_config_accepts_native_and_rejects_unknown():
    E2cModelConfig(map_pyramid_source="native_hierarchical").validate()
    with pytest.raises(ValueError):
        E2cModelConfig(map_pyramid_source="bogus").validate()


def test_resolved_config_carries_mode():
    """The checkpoint's resolved_config records the map_pyramid_source (auditable off-ckpt)."""
    cfg = E2cModelConfig(agg="supervlad", pyramid_mode="cell",
                         map_pyramid_source="native_hierarchical")
    assert cfg.to_dict()["map_pyramid_source"] == "native_hierarchical"
