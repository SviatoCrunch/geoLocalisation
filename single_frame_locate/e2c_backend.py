"""E2c (geo_iso) backend adapter — SuperVLAD/residual concentric-pyramid model.

This is the model family trained by ``geoLocalisation/geo_e2c_train`` (checkpoint
schema ``{"model": state_dict, "resolved_config": {...}, "epoch": N}`` — NOT the
``config``/``state_dict`` schema the Stage-1/2 adapters expect, which is why the
auto stage detector misreads it as Stage-1). The model itself is ``siam_e2c_model``'s
``E2cModel`` (query-conditioned SuperVLAD cell pyramid), whose SuperVLAD assignment
weights are baked INTO the checkpoint — no external ``c_centers.pt`` vocabulary is
needed (unlike Stage-1). It does still load the recast assign dict (``k32.pt``) at
build time; its path comes from the checkpoint's ``resolved_config.assign_path`` but
that is a dev-box path, so override it with ``SIAM_ASSIGN_PATH`` on the server.

Query/map token PARITY: both the map gallery H5 and the live query frame must be
DINO-extracted + projected + L2-normalised identically. The gallery was built by
``map_extract`` (``--backbone dinov2_vitg14 --desc_dim 1024 --proj_seed 0``); this
adapter reuses the SAME ``map_extract`` primitives for the query so the two token
spaces line up. Wrong ``desc_dim``/``proj_seed`` would load fine but retrieve garbage.

Env knobs (all optional except SIAM_ASSIGN_PATH when the config path is absent):
    SIAM_ASSIGN_PATH   recast k32.pt (overrides resolved_config.assign_path)
    SIAM_E2C_DESC_DIM  projected token dim (default: cfg.d_token, i.e. 1024)
    SIAM_E2C_PROJ_SEED RandomProjector seed (default: 0 — MUST match the gallery)
    SIAM_BACKBONE / SIAM_DINO_FACET  DINO overrides (default: dinov2_vitg14 / value)
    SIAM_NO_SKY=1      disable sky filtering on the query
    SIAM_MAX_SIDE      cap query long side before DINO
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

# Standalone (geoLocalisation) vendoring of the server's stage_e2c_adapter: the ONLY change
# vs the byte-for-byte server file is this path bootstrap. siam_e2c_model + map_extract live
# directly under geoLocalisation/ (this file's grand-parent), so putting that on sys.path makes
# the e2c model + its map_extract extraction helpers import EXACTLY as on the server.
_GEO_DIR = Path(__file__).resolve().parent.parent


def _ensure_sys_path() -> None:
    for p in (str(_GEO_DIR),):
        if p not in sys.path:
            sys.path.insert(0, p)


def is_e2c_checkpoint(ckpt: str) -> bool:
    """True when the checkpoint is the geo_e2c schema (resolved_config + model),
    NOT the Stage-1/2 config/state_dict schema. Cheap: reads only the pickle top level."""
    try:
        obj = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        try:
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        except Exception:
            return False
    if not isinstance(obj, dict):
        return False
    rc = obj.get("resolved_config")
    ok = bool(rc) and "model" in obj and isinstance(rc, dict) and "agg" in rc
    del obj
    return ok


class E2cBackend:
    # Stage number is only used for logging; the e2c model is a Stage-2 relative
    # (query-conditioned SuperVLAD) so report 2.
    stage = 2

    def __init__(self, ckpt: str, device, env):
        _ensure_sys_path()
        from siam_e2c_model.config import E2cModelConfig
        from siam_e2c_model.model import build_e2c_model
        from map_extract.dino import build_dino_extractor, RandomProjector

        checkpoint = torch.load(Path(ckpt), map_location="cpu", weights_only=False)
        rc = checkpoint.get("resolved_config")
        state = checkpoint.get("model")
        if not rc or state is None:
            raise RuntimeError(
                f"checkpoint {ckpt} is not an e2c checkpoint "
                f"(need 'resolved_config' + 'model' keys).")

        # resolved_config → E2cModelConfig (only the dataclass fields; the config
        # also carries data-pipeline keys like concentric_sizes_m we ignore here).
        keep = {f.name for f in fields(E2cModelConfig)}
        cfg_kwargs = {k: rc[k] for k in rc if k in keep}

        # The baked assign_path points at the training box; override on the server.
        assign_path = env("SIAM_ASSIGN_PATH") or cfg_kwargs.get("assign_path") or ""
        if not assign_path or not os.path.exists(assign_path):
            raise RuntimeError(
                f"e2c assign dict not found (assign_path={assign_path!r}). "
                f"Download k32.pt and set SIAM_ASSIGN_PATH to it.")
        cfg_kwargs["assign_path"] = assign_path

        cfg = E2cModelConfig(**cfg_kwargs)
        model = build_e2c_model(cfg, device=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning("e2c load_state_dict non-strict  missing=%s unexpected=%s",
                        list(missing)[:8], list(unexpected)[:8])
        model.eval()

        # ── query-side DINO + projector (reuse map_extract for exact parity) ──
        backbone = env("SIAM_BACKBONE") or "dinov2_vitg14"
        facet = env("SIAM_DINO_FACET", "value")
        desc_dim = int(env("SIAM_E2C_DESC_DIM") or cfg.d_token)     # 1024
        proj_seed = int(env("SIAM_E2C_PROJ_SEED") or 0)            # MUST match gallery
        dino = build_dino_extractor(backbone, layer=None, facet=facet,
                                    device=device, norm_descs=False)
        self.patch_size = getattr(dino, "patch_size", 14)
        self._preprocess = getattr(dino, "preprocess", None)
        self.projector = RandomProjector(desc_dim, proj_seed)

        sky_seg = None
        if env("SIAM_NO_SKY", "0") not in ("1", "true", "True"):
            from map_extract.sky import SkySegmenter
            sky_seg = SkySegmenter()

        self.model = model
        self.dino = dino
        self.cfg = cfg.to_dict()
        self.scales_cells = list(cfg.scales_cells)
        self.d_token = int(cfg.d_token)
        self.sky_seg = sky_seg
        self.tile_size_m = (float(env("SIAM_TILE_SIZE_M")) if env("SIAM_TILE_SIZE_M")
                            else float(rc.get("tile_size_m", 1000.0)))
        self.device = device
        self.max_side = int(env("SIAM_MAX_SIDE")) if env("SIAM_MAX_SIDE") else None
        self._env = env
        self._ckpt = ckpt
        self._amp = env("SIAM_AMP", "0") in ("1", "true", "True")
        log.info("e2c backend ready  agg=%s k=%s d_token=%d scales=%s assign=%s",
                 cfg.agg, cfg.k, self.d_token, self.scales_cells, assign_path)

    @property
    def default_index_cache(self) -> str:
        vlad_cache = self._env("SIAM_VLAD_CACHE")
        base = vlad_cache if vlad_cache else str(Path(self._ckpt).parent)
        return os.path.join(base, "siam_e2c_map_V.pt")

    # ── query: RGB frame → projected+normalised token GRID (tokens + patch coords) ──
    @torch.no_grad()
    def query_grid(self, rgb: np.ndarray):
        """RGB frame → (tokens (N,D) on device, xy (N,2) np patch coords). Sky-filtered, projected,
        L2-space-matched to the gallery. The raw token grid the fine reranker mutual-NN needs; xy is
        row-major (x=col, y=row) to match the map-cell grids."""
        _ensure_sys_path()
        from map_extract.extract import _tile_to_batch, _extract_level_features

        if self.max_side and max(rgb.shape[:2]) > self.max_side:
            s = self.max_side / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        # Mirror map_extract.extract_images EXACTLY: feed the raw RGB frame to
        # _tile_to_batch (its _PREPROCESS owns the resize/normalise) — no pre-resize,
        # so the query token space matches how query_kup_d1024.h5 was built.
        batch, h_r, w_r = _tile_to_batch([rgb], self.device, self.patch_size, self._preprocess)
        feats = _extract_level_features(self.dino, batch, h_r, w_r, self.projector, amp=self._amp)
        feat = feats[0]
        if feat.ndim == 4 and feat.shape[0] == 1:
            feat = feat[0]                                   # (D, H, W)
        D, ph, pw = feat.shape
        flat = feat.reshape(D, ph * pw)                      # (D, N)

        keep = np.ones(ph * pw, dtype=bool)
        if self.sky_seg is not None:
            from map_extract.sky import sky_keep_mask
            mask = self.sky_seg.segment(rgb, threshold=0.5)
            sk = np.asarray(sky_keep_mask(mask, (ph, pw), 0.5))
            if sk.any():
                keep = sk
        ys, xs = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
        xy = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)[keep]   # (N,2) col,row
        tokens = torch.from_numpy(flat[:, keep].T).contiguous().to(self.device).float()  # (N, D)
        return tokens, xy

    @torch.no_grad()
    def encode_query_full(self, rgb: np.ndarray):
        """→ (query embedding, tokens (N,D), xy (N,2)) — ONE DINO forward for both coarse + fine."""
        tokens, xy = self.query_grid(rgb)
        return self.model.encode_query(tokens), tokens, xy

    @torch.no_grad()
    def encode_query(self, rgb: np.ndarray) -> torch.Tensor:
        return self.encode_query_full(rgb)[0]

    # ── score: query embedding vs cached cell-pyramid V ───────────────────────
    def score(self, q: torch.Tensor, V: dict, chunk: int) -> np.ndarray:
        Q = q.unsqueeze(0) if q.dim() == 1 else q
        M = V[self.scales_cells[0]].shape[0]
        sims = np.empty(M, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for s in range(0, M, chunk):
                e = min(M, s + chunk)
                Vc = {n: V[n][s:e].to(self.device) for n in self.scales_cells}
                sc = self.model.score(Q, Vc)                 # (1, e-s)
                sims[s:e] = sc[0].detach().cpu().numpy()
        return sims

    # ── gallery: precomputed d1024 token-grid H5 → cell pyramid V ─────────────
    def build_gallery_h5(self, map_h5: str, chunk: int) -> tuple:
        import h5py

        with h5py.File(map_h5, "r") as f:
            keys = list(f.keys())
            lat = np.array([float(f[k].attrs.get("lat", np.nan)) for k in keys])
            lon = np.array([float(f[k].attrs.get("lon", np.nan)) for k in keys])
            win = np.array([float(f[k].attrs.get("window_size_m", self.tile_size_m))
                            for k in keys])
            parts = {n: [] for n in self.scales_cells}
            self.model.eval()
            with torch.no_grad():
                for s in range(0, len(keys), chunk):
                    grids = []
                    for k in keys[s:s + chunk]:
                        arr = np.asarray(f[k]["ift_dino"])
                        if arr.ndim == 4 and arr.shape[0] == 1:
                            arr = arr[0]                     # (D, H, W)
                        if arr.shape[0] != self.d_token:
                            raise RuntimeError(
                                f"{k} ift_dino d={arr.shape[0]} != checkpoint "
                                f"d_token={self.d_token} — gallery must be extracted at "
                                f"desc_dim={self.d_token} (backbone/proj mismatch).")
                        D, H, W = arr.shape
                        g = torch.from_numpy(
                            arr.reshape(D, H * W).T.reshape(H, W, D).astype(np.float32))
                        grids.append(g)
                    grids_t = torch.stack(grids).to(self.device)   # (m, H, W, D)
                    Vc = self.model.build_V(grids_t)
                    for n in self.scales_cells:
                        parts[n].append(Vc[n].cpu())
                    del grids_t, Vc
        V = {n: torch.cat(parts[n]) for n in self.scales_cells}
        return V, keys, lat, lon, win

    def build_gallery_tif(self, tif: str, env) -> tuple:
        raise RuntimeError(
            "e2c backend has no on-the-fly TIF gallery; provide a precomputed "
            "d1024 token H5 via SIAM_MAP_H5 (map_extract --desc_dim 1024 --proj_seed 0).")
