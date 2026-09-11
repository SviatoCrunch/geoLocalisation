"""Residual VLAD in concentric mode — explicit reference for the pooling ORDER.

Policy = Variant A (full support). This independently recomputes the documented order
    hard assignment → residual → fractional spatial weighting → per-cluster sum → normalization
and checks it matches ``VladAggregation.build_V`` on the concentric core, so the residual arm
cannot silently fall back to cell math / bypass the fractional masks.
"""
import pytest
import torch
import torch.nn.functional as F

from ._synth import build, grids


@pytest.mark.parametrize("agg", ["vlad", "residual"])
def test_concentric_residual_matches_manual_reference(agg):
    m, _aw, centroids = build(agg, k=6, d=64, d_out=32, d_group=8, pyramid_mode="concentric",
                              concentric_sizes_m=(1000.0, 500.0, 250.0))
    m.eval()
    core = m.core
    C = F.normalize(torch.as_tensor(centroids).float(), dim=1)          # (K,D) as the arm stores
    g = grids(m=3, g=6, d=64)
    M, H, W, D = g.shape
    eps = core.eps
    flat = F.normalize(g.reshape(M, H * W, D).float(), dim=-1, eps=eps)
    labels = torch.cdist(flat, C.unsqueeze(0).expand(M, -1, -1)).argmin(-1)   # (M,N)
    res = flat - C[labels]                                              # residual
    K = C.shape[0]
    gp, mh = core.group_proj, core.map_head
    masks = core.masks(H, W, g.device, res.dtype)                      # (L,N) fractional

    V_model = m.build_V(g)
    for li in core.scales_cells:
        w = masks[li].view(1, -1, 1)                                   # fractional spatial weight
        block = torch.zeros(M, K, D)
        block.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, D), res * w)   # weight → sum
        block = block.view(M, 1, K, D)
        if core.intra:
            block = F.normalize(block, dim=3, eps=eps)
        block = F.normalize(gp(block), dim=3, eps=eps).reshape(M, 1, K * gp.d_out)
        region = F.normalize(block, dim=2, eps=eps)
        V_ref = mh(region)                                             # map head + final L2
        assert torch.allclose(V_ref, V_model[li], atol=1e-5)


def test_out_of_crop_tokens_do_not_contribute():
    # a token with zero fractional weight at the apex must not change that level's descriptor
    m, _, _ = build("residual", k=6, d=64, d_out=16, pyramid_mode="concentric",
                    concentric_sizes_m=(1000.0, 500.0, 250.0))
    m.eval()
    g = grids(m=1, g=6, d=64)
    specs = m.core.mask_specs(6, 6)
    apex = specs[-1]["weight"].reshape(6, 6)                           # 250 m level
    corner = (0, 0)
    assert apex[corner] == 0.0                                         # corner is outside the apex crop
    V0 = m.build_V(g)
    g2 = g.clone(); g2[0, corner[0], corner[1], :] = 999.0             # perturb an out-of-crop token
    V1 = m.build_V(g2)
    apex_key = m.core.scales_cells[-1]
    assert torch.allclose(V0[apex_key], V1[apex_key], atol=1e-5)       # apex unchanged
