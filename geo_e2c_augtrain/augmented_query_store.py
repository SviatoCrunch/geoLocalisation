"""Per-epoch AUGMENTED query token store - the drop-in replacement for geo_e2c_train.QueryTokenStore
that makes real drone-image augmentation possible.

`tokens(qid)` returns FRESH tokens for the current epoch:
    raw drone frame -> uav_augment (vendored stage3 recipe) -> sky mask (transformed with the frame)
    -> DINO (map_extract extractor) -> SAME RandomProjector (seed/backbone read from the reference
       query H5) -> L2-norm -> drop sky tokens -> (N, D)
so the augmented query lives in the EXACT space of the frozen gallery (same seed-0 JL projection).

`set_epoch(e)` re-encodes all requested queries once (batched by frame shape) into a cache; `tokens`
then serves from that cache. Augmentation is deterministic per (base_seed, epoch, qid) and TRAIN-ONLY
(val/test evaluate with the frozen query H5 via the normal QueryTokenStore). Interface matches
QueryTokenStore: `has` / `ids` / `tokens` (+ `set_epoch`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _scan(root) -> dict:
    out = {}
    for p in sorted(Path(root).expanduser().rglob("*")):
        if p.suffix.lower() in _IMG_EXTS:
            out.setdefault(p.stem, p)
    return out


def _wid(qid: str) -> int:
    return int.from_bytes(hashlib.sha256(qid.encode()).digest()[:4], "big")


class AugmentedQueryStore:
    def __init__(self, gt_by_city: dict, ref_query_h5, device="cuda", preset="stage3_strong",
                 segment_sky=True, base_seed=0, batch=8, amp=False,
                 sky_pixel_threshold=0.5, sky_patch_threshold=0.5):
        import h5py
        from map_extract.dino import build_dino_extractor, RandomProjector
        from .uav_augment import AugConfig

        with h5py.File(Path(ref_query_h5).expanduser(), "r") as f:            # proj params == the gallery
            backbone = str(f.attrs.get("backbone", "dinov2_vitg14"))
            desc_dim = int(f.attrs.get("projection_out_dim", 1024))
            seed = int(f.attrs.get("projection_seed", 0))
            facet = str(f.attrs.get("dino_facet", "value"))
        self.device, self.batch, self.amp, self.base_seed = device, batch, amp, base_seed
        self.sky_pixel_threshold, self.sky_patch_threshold = sky_pixel_threshold, sky_patch_threshold
        self.dino = build_dino_extractor(backbone, facet=facet, device=device, norm_descs=False)
        self.patch_size = int(getattr(self.dino, "patch_size", 14))
        self.preprocess = getattr(self.dino, "preprocess", None)
        self.projector = RandomProjector(desc_dim, seed)
        self.sky = None
        if segment_sky:
            from map_extract.sky import SkySegmenter
            self.sky = SkySegmenter()
        self.cfg = AugConfig.from_cfg({"enabled": True, "preset": preset})
        self._index = {}                                                     # "city:stem" -> image path
        for city, root in gt_by_city.items():
            for stem, p in _scan(root).items():
                self._index[f"{city}:{stem}"] = p
        self._skycache, self._tok, self.epoch = {}, {}, -1
        print(f"[augstore] backbone={backbone} desc_dim={desc_dim} seed={seed} facet={facet} "
              f"preset={preset} sky={'yes' if self.sky else 'no'} frames={len(self._index)}", flush=True)
        for line in self.cfg.summary_lines():
            print("   " + line, flush=True)

    def has(self, qid) -> bool:
        return qid in self._index

    def ids(self) -> set:
        return set(self._index)

    def _base_sky(self, qid, rgb):
        if self.sky is None:
            return None
        m = self._skycache.get(qid)
        if m is None:
            m = self.sky.segment(rgb, threshold=self.sky_pixel_threshold)      # segment ONCE per frame
            self._skycache[qid] = m
        return m

    def set_epoch(self, epoch: int, qids=None, progress=False):
        """Re-encode the requested queries for this epoch (augmented). Call before tokens()."""
        from PIL import Image
        from map_extract.extract import _tile_to_batch, _extract_level_features
        from map_extract.sky import sky_keep_mask
        from .uav_augment import augment_frame, make_aug_rng
        self.epoch, self._tok = epoch, {}
        ids = [q for q in (qids if qids is not None else self._index) if q in self._index]
        it = ids
        if progress:
            from tqdm import tqdm
            it = tqdm(ids, desc=f"aug re-encode e{epoch}", unit="q")
        aug = []                                                              # (qid, aug_rgb, aug_mask)
        for qid in it:
            rgb = np.array(Image.open(self._index[qid]).convert("RGB"))
            sky = self._base_sky(qid, rgb)
            rng = make_aug_rng(self.base_seed, epoch, worker_id=_wid(qid))
            a_rgb, a_mask, _, _ = augment_frame(rgb, sky, self.cfg, rng)
            aug.append((qid, a_rgb, a_mask))
        groups = {}                                                          # same shape -> one DINO batch
        for qid, r, m in aug:
            groups.setdefault(r.shape, []).append((qid, r, m))
        for _, grp in groups.items():
            for s in range(0, len(grp), self.batch):
                chunk = grp[s:s + self.batch]
                batch_t, h_r, w_r = _tile_to_batch([r for _, r, _ in chunk], self.device,
                                                   self.patch_size, self.preprocess)
                feats = _extract_level_features(self.dino, batch_t, h_r, w_r, self.projector, amp=self.amp)
                for (qid, _, m), feat in zip(chunk, feats):
                    f = feat[0] if (feat.ndim == 4 and feat.shape[0] == 1) else feat   # (D,ph,pw)
                    D, ph, pw = f.shape
                    flat = f.reshape(D, ph * pw)
                    if m is not None:
                        keep = sky_keep_mask(m, (ph, pw), self.sky_patch_threshold)
                        if int(keep.sum()) > 0:
                            flat = flat[:, keep]
                    self._tok[qid] = torch.from_numpy(flat.T.astype(np.float32))       # (N,D)

    def tokens(self, qid) -> torch.Tensor:
        return self._tok[qid]

    def close(self):
        self._tok, self._skycache = {}, {}
