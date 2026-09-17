"""Tail parity + no-new-params for the native map-pyramid source.

These are the load-bearing tests: they prove that building V from FROZEN per-cell per-group
sums (the native source's data path) reproduces the legacy ``build_V`` bit-for-bit, and that the
model tail (params / state-dict) is identical between the two sources.

Run:  cd geoLocalisation && python -m pytest native_map_pyramid/tests/test_tail_parity.py -q
"""
import math

import torch
import torch.nn.functional as F

from siam_e2c_model.vendored.stage2_core import (
    build_stage2_query_conditioned_model,
    cell_group_sums_from_grid,
    tokens_to_group_sum,
    pyramid_from_cell_group_sums,
)


def _model(seed=0, d_token=64, K=8, dg=8, scales=(8, 4, 2, 1), d_out=32, dropout=0.0):
    """Small Stage-2 cell/supervlad model with a fixed frozen assignment (dropout off)."""
    torch.manual_seed(seed)
    assign = F.normalize(torch.randn(K, d_token), dim=1)
    m = build_stage2_query_conditioned_model(
        d_token=d_token, n_groups=K, n_ghost=0, group_projection_dim=dg,
        scales_cells=scales, d_out=d_out, head_hidden=64, dropout=dropout,
        assign_weight=assign, freeze_assignment=True)
    m.eval()
    return m


def test_build_V_from_cell_sums_matches_legacy_build_V():
    """S from the SAME grid via cell_group_sums_from_grid → build_V_from_cell_sums == build_V."""
    torch.manual_seed(1)
    m = _model()
    M, H, W, D = 3, 24, 24, 64          # 24 divisible by 8 → valid n=8 cells (3 tokens/cell)
    grids = torch.randn(M, H, W, D)

    with torch.no_grad():
        V_legacy = m.build_V(grids)
        S = cell_group_sums_from_grid(grids, m.agg, m.scales_cells)
        V_native = m.build_V_from_cell_sums(S)

    for n in m.scales_cells:
        assert V_legacy[n].shape == (M, n * n, 32)
        assert V_native[n].shape == (M, n * n, 32)
        assert torch.allclose(V_legacy[n], V_native[n], atol=1e-5, rtol=1e-4), \
            f"scale n={n} mismatch: max|Δ|={(V_legacy[n]-V_native[n]).abs().max():.2e}"


def test_final_descriptors_are_l2_normalized():
    m = _model()
    grids = torch.randn(2, 16, 16, 64)
    with torch.no_grad():
        S = cell_group_sums_from_grid(grids, m.agg, m.scales_cells)
        V = m.build_V_from_cell_sums(S)
    for n, Vn in V.items():
        norms = Vn.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), n


def test_tile_chunking_is_numerically_identical():
    m = _model()
    grids = torch.randn(5, 16, 16, 64)
    with torch.no_grad():
        S = cell_group_sums_from_grid(grids, m.agg, m.scales_cells)
        V_full = m.build_V_from_cell_sums(S)
        V_chunk = m.build_V_from_cell_sums(S, tile_chunk=2)
    for n in m.scales_cells:
        assert torch.allclose(V_full[n], V_chunk[n], atol=1e-6), n


def test_tokens_to_group_sum_matches_partition_of_single_cell():
    """The per-crop aggregator (native n=1) equals the n=1 slice of cell_group_sums_from_grid."""
    m = _model()
    grid = torch.randn(1, 16, 16, 64)
    with torch.no_grad():
        S = cell_group_sums_from_grid(grid, m.agg, [1])          # (1,1,K,Dv)
        s_single = tokens_to_group_sum(grid[0], m.agg)           # (K,Dv)
    assert torch.allclose(S[1][0, 0], s_single, atol=1e-5)


def test_native_source_adds_no_parameters_and_same_state_dict():
    """build_V_from_cell_sums reuses the same modules → identical params / state_dict keys."""
    m = _model()
    trainable_before = sum(p.numel() for p in m.parameters() if p.requires_grad)
    sd_keys = set(m.state_dict().keys())

    grids = torch.randn(2, 16, 16, 64)
    with torch.no_grad():
        S = cell_group_sums_from_grid(grids, m.agg, m.scales_cells)
        _ = m.build_V_from_cell_sums(S)          # exercising the native path adds nothing

    trainable_after = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert trainable_after == trainable_before
    assert set(m.state_dict().keys()) == sd_keys
    # tail modules present and trainable
    assert m.group_proj.weight.requires_grad
    assert all(p.requires_grad for p in m.map_head.parameters())
    assert not m.agg.assign.weight.requires_grad          # assignment stays frozen


def test_gradients_reach_the_tail_but_not_the_frozen_assignment():
    """Backward through the native path trains group_proj/map_head/scale_gate/rho, not assign."""
    m = _model(dropout=0.0)
    m.train()
    grids = torch.randn(2, 16, 16, 64)
    q = F.normalize(torch.randn(2, 32), dim=-1)          # (B, d_out)
    S = cell_group_sums_from_grid(grids, m.agg, m.scales_cells)
    V = m.build_V_from_cell_sums(S)
    scores = m.score_queries_against_tiles(q, V)          # (B, M)
    loss = scores.sum()
    loss.backward()

    assert m.group_proj.weight.grad is not None and torch.isfinite(m.group_proj.weight.grad).all()
    assert m.map_head.net[0].weight.grad is not None
    assert m.scale_gate.net[0].weight.grad is not None
    assert m.rho[str(8)].grad is not None
    assert m.agg.assign.weight.grad is None              # frozen → no grad
