"""MapPyramidSource — the Strategy dispatched at the two map-pyramid build points
(``geo_e2c_train/train.py`` training loop and ``geo_e2c_train/eval.py`` gallery pass).

Both strategies return the SAME ``V {n:(M,n²,d_out)}`` the model expects; only the SOURCE of the
per-cell features differs:

  * :class:`LegacyTokenPartitionSource` — token grids from the H5, partitioned into n×n cells
    (``model.build_V``). Byte-identical to the pre-existing path.
  * :class:`NativeHierarchicalSource` — frozen per-cell per-group sums from the native ``S`` cache,
    run through the same trainable tail (``model.build_V_from_cell_sums``).

Neither adds trainable parameters.
"""
from __future__ import annotations

from . import cache as C


class LegacyTokenPartitionSource:
    name = "legacy_token_partition"

    def __init__(self, tile_loader, device):
        self.tile_loader = tile_loader
        self.device = device

    def build_V(self, model, tile_ids):
        G = self.tile_loader.stack(list(tile_ids)).float().to(self.device)
        return model.build_V(G)


class NativeHierarchicalSource:
    name = "native_hierarchical"

    def __init__(self, native_cache: "C.NativeCellCache", device):
        self.cache = native_cache
        self.device = device

    def build_V(self, model, tile_ids):
        S = self.cache.load(list(tile_ids))                     # {n:(m,n²,K,Dv)} fp32
        return model.build_V_from_cell_sums({n: S[n].to(self.device) for n in S})


def _patch_size_for(backbone: str) -> int:
    return 16 if str(backbone).startswith("dinov3") else 14


def build_expected_identity(*, assign_weight, k: int, d_token: int, backbone: str,
                            dino_layer, dino_facet: str, output_px: int, tile_size_m: float,
                            proj_seed: int, proj_in_dim: int) -> dict:
    """The full native-cache fingerprint the CURRENT run expects. Compared field-by-field against
    the cache's stored identity (safety-critical: ``vlad_dict_id`` must match the model's frozen
    assignment, ``d_value``/``projection_out_dim`` must equal ``d_token``)."""
    projection = "random_gaussian_jl" if d_token < proj_in_dim else "none"
    return C.build_fingerprint(
        backbone=backbone, dino_layer=dino_layer, dino_facet=dino_facet, output_px=output_px,
        tile_size_m=tile_size_m, patch_size=_patch_size_for(backbone), projection=projection,
        projection_seed=proj_seed, projection_in_dim=proj_in_dim, projection_out_dim=d_token,
        n_groups=k, d_value=d_token, vlad_dict_id=C.vlad_dict_id(assign_weight))


def open_native_cache(path, expected_identity: dict, *, preload: bool = True) -> "C.NativeCellCache":
    """Open + STRICTLY validate a native cache; raises :class:`cache.CacheIncompatibleError`.

    ``preload=True`` (default for training) loads all ``S`` into RAM once so per-batch reads are
    instant (parity with the legacy TileGridLoader RAM cache)."""
    cache = C.NativeCellCache(path, preload=preload)
    cache.assert_compatible(expected_identity)
    return cache
