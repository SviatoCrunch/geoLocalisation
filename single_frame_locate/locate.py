"""Single-frame geo-localization — faithful standalone of the PROD service GPU path.

This reproduces, for ONE drone frame, exactly what the live vision_project service
(`server/e2c-live-snapshot`, service/siam_tools/siam.py `SiamEngine.localize`) runs on the GPU:

  Stage A — COARSE (e2c SuperVLAD), GPU
    encode_query_full(rgb): one DINOv2-vitg14 forward → token grid → RandomProjector(d=1024,
    seed=0) → sky filter → (tokens (N,D), xy) + query embedding q.
    score(q, V, chunk): SuperVLAD cell-pyramid score of q vs the cached gallery V
    (built once from the map token H5). argsort(-sims)[:K] → coarse top-K candidate cells.

  Stage B — FINE rerank (GPU MAGSAC++), GPU
    rerank_multiframe([tokens], [xy], cell_ids, store): packed /features pyramid block per cell →
    fp16 batched match Q@blockᵀ over all (pos×level) grids → mutual-NN → ONE batched
    estimate_homography_magsacpp_batch over all crops → inlier counts.
    level_agg='sum' over levels → cell_agg='mean' over positions; best position = point,
    best level = level_m. → top-K distinct cells.

Byte-for-byte vendored from the server: ``rerank/`` (matcher, reranker, magsacpp_torch, newformat)
and ``e2c_backend.py`` (= service stage_e2c_adapter, only the sys.path bootstrap changed). The ONLY
new code is this orchestrator, which is the single-frame slice of ``SiamEngine`` with the FastAPI /
SQS / pair-render / GPU-lock glue stripped — the compute is identical.

Example (kram):
    python -m single_frame_locate.locate \
        --image frame.jpg \
        --ckpt   /path/run_supervlad_cell_w1_best.pt \
        --map-h5 /path/kram20_prodaction.h5 \
        --assign /path/k32.pt \
        --store  /path/pyramid_newformat/kram \
        --city kram --k 100 --topk 10 --device cuda --out result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Allow both ``python -m single_frame_locate.locate`` and ``python locate.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from single_frame_locate.e2c_backend import E2cBackend
else:
    from .e2c_backend import E2cBackend

_METERS_PER_DEG_LAT = 110540.0            # same constant as the service _square_ring


def _square_ring(lat: float, lon: float, half_m: float) -> list:
    """Closed lon/lat ring of a square tile centred on (lat, lon) — identical to the service helper."""
    dlat = half_m / _METERS_PER_DEG_LAT
    dlon = half_m / (_METERS_PER_DEG_LAT * max(np.cos(np.radians(lat)), 1e-6))
    return [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat],
            [lon + dlon, lat + dlat], [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]


def _resolve_device(spec: str):
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return spec


class SingleFrameLocator:
    """Lazy load of the e2c coarse model + gallery V + packed pyramid store; localize one frame.

    Mirrors SiamEngine: owns the gallery cache (V, keys, lat/lon/win), the coarse → fine wiring, and
    the reranked-cell → match-dict mapping. No AOI filter / pair render / GPU lock (single-process
    offline tool), so the GPU compute is the only thing that runs — exactly as production."""

    def __init__(self, ckpt: str, map_h5: str, assign_path: str, *, store: str,
                 device: str = "auto", city: str = "kram", rerank_k: int = 100,
                 rerank_topk: int = 10, rerank_reproj: float = 2.0, chunk: int = 256,
                 index_cache: str | None = None, rebuild: bool = False,
                 store_format: str = "packed"):
        self.device = _resolve_device(device)
        # the batched GPU MAGSAC++ reads its device from this env var (reranker._mpp_config);
        # pin it to the same device so coarse + fine run together on the GPU, as on the server.
        os.environ.setdefault("SIAM_RERANK_DEVICE", str(self.device))

        def _env(k, d=None):
            if k == "SIAM_ASSIGN_PATH" and assign_path:
                return assign_path
            return os.environ.get(k, d)

        self.backend = E2cBackend(ckpt, self.device, _env)
        self.scales_cells = self.backend.scales_cells
        self.tile_size_m = self.backend.tile_size_m
        self.chunk = int(chunk)
        self.city = city
        self.rerank_k = int(rerank_k)
        self.rerank_topk = int(rerank_topk)
        self.rerank_reproj = float(rerank_reproj)
        self.store_path = store
        self.store_format = store_format
        self._store = None

        self._build_or_load_gallery(map_h5, index_cache, rebuild)
        self.half_m = (float(np.median(self.win[self.win > 0])) / 2.0
                       if (self.win > 0).any() else self.tile_size_m / 2.0)

    # -- gallery V (same cache contract as SiamEngine._build_or_load_gallery_h5) --
    def _build_or_load_gallery(self, map_h5: str, index_cache: str | None, rebuild: bool) -> None:
        if index_cache and os.path.exists(index_cache) and not rebuild:
            blob = torch.load(index_cache, map_location="cpu", weights_only=False)
            if blob.get("map_h5") == str(map_h5) and blob.get("ckpt") == str(self.backend._ckpt):
                self.V, self.keys = blob["V"], blob["keys"]
                self.lat, self.lon, self.win = blob["lat"], blob["lon"], blob["win"]
                return
        (self.V, self.keys, self.lat, self.lon, self.win) = \
            self.backend.build_gallery_h5(map_h5, self.chunk)
        if index_cache:
            torch.save({"V": self.V, "keys": self.keys, "lat": self.lat, "lon": self.lon,
                        "win": self.win, "map_h5": str(map_h5),
                        "ckpt": str(self.backend._ckpt)}, index_cache)

    def _load_store(self):
        if self._store is None:
            if self.store_format == "packed":
                from .rerank.newformat import NewFormatCellStore
                self._store = NewFormatCellStore(self.store_path)
            else:
                from .rerank import CellStore
                self._store = CellStore(self.store_path)
        return self._store

    # -- coarse: one DINO forward → top-K candidate cells + the query token grid --
    @torch.no_grad()
    def _topk_full(self, rgb: np.ndarray, top: int):
        q, tokens, xy = self.backend.encode_query_full(rgb)
        sims = self.backend.score(q, self.V, self.chunk)
        order = np.argsort(-sims)[:top]
        return [(int(i), float(sims[i])) for i in order], tokens, xy

    # -- fine: GPU MAGSAC++ rerank of the candidate cells (single frame = F=1 list) --
    def _rerank(self, tokens, xy, cand_tis: list[int], topk: int) -> list[dict]:
        from .rerank import rerank_multiframe
        store = self._load_store()
        cell_ids = [f"{self.city}:{self.keys[ti]}" for ti in cand_tis]
        return rerank_multiframe([tokens.detach()], [np.asarray(xy, np.float64)], cell_ids, store,
                                 reproj_thresh=self.rerank_reproj, topk=topk,
                                 level_agg="sum", cell_agg="mean")

    def _match_reranked(self, rec: dict, rank: int) -> dict:
        lat, lon = float(rec["lat"]), float(rec["lon"])
        half = float(rec["level_m"]) / 2.0
        cid = rec["cell_id"]
        return {"rank": rank, "tile_key": cid, "image_number": str(cid),
                "coordinates": {"lat": lat, "lon": lon},
                "similarity": round(float(rec["cell_score"]), 6),
                "level_m": float(rec["level_m"]),
                "geometry": {"type": "Polygon",
                             "coordinates": [_square_ring(lat, lon, half)]}}

    # -- public: one frame → top-K reranked matches (SiamEngine.localize, single-frame slice) --
    def localize(self, img_bgr: np.ndarray, *, top: int | None = None) -> list[dict]:
        top = int(top or self.rerank_topk or 10)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        coarse, tokens, xy = self._topk_full(rgb, self.rerank_k)
        recs = self._rerank(tokens, xy, [ti for ti, _ in coarse], self.rerank_topk or top)
        return [self._match_reranked(rec, r) for r, rec in enumerate(recs, 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", action="append", default=[],
                    help="query drone frame (repeatable). Omit to use --images-dir.")
    ap.add_argument("--images-dir", default=None,
                    help="directory of query frames — localize ALL images in it (jpg/jpeg/png, recursive).")
    ap.add_argument("--ckpt", required=True, help="e2c checkpoint (resolved_config + model)")
    ap.add_argument("--map-h5", required=True, help="map token gallery H5 (map_extract d1024)")
    ap.add_argument("--assign", required=True, help="recast assign dict k32.pt (SIAM_ASSIGN_PATH)")
    ap.add_argument("--store", required=True, help="packed /features pyramid store dir or s3://…/ ")
    ap.add_argument("--city", default="kram", help="cell-id namespace (gallery key N_lvl0 → city:N_lvl0)")
    ap.add_argument("--k", type=int, default=100, help="coarse candidate pool width (SIAM_RERANK_K)")
    ap.add_argument("--topk", type=int, default=10, help="final reranked cell count")
    ap.add_argument("--reproj", type=float, default=2.0, help="MAGSAC inlier reproj threshold (px)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    ap.add_argument("--store-format", default="packed", choices=["packed", "percell"])
    ap.add_argument("--index-cache", default=None, help="optional path to cache/reuse the gallery V")
    ap.add_argument("--rebuild", action="store_true", help="force-rebuild the gallery V cache")
    ap.add_argument("--out", default=None,
                    help="write the combined JSON (list of per-image results) here (default: stdout)")
    ap.add_argument("--out-dir", default=None,
                    help="also write one <stem>.json per image into this dir")
    args = ap.parse_args(argv)

    images = list(args.image)
    if args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        images += [str(p) for p in sorted(Path(args.images_dir).rglob("*")) if p.suffix in exts]
    images = list(dict.fromkeys(images))                 # dedup, keep order
    if not images:
        raise SystemExit("no query frames — pass --image <f> (repeatable) and/or --images-dir <dir>")

    # load the model + gallery + store ONCE, then localize every frame
    loc = SingleFrameLocator(args.ckpt, args.map_h5, args.assign, store=args.store,
                             device=args.device, city=args.city, rerank_k=args.k,
                             rerank_topk=args.topk, rerank_reproj=args.reproj,
                             index_cache=args.index_cache, rebuild=args.rebuild,
                             store_format=args.store_format)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for n, path in enumerate(images, 1):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{n}/{len(images)}] SKIP unreadable {path}")
            results.append({"image": path, "error": "unreadable"})
            continue
        try:
            matches = loc.localize(img)
        except Exception as e:                           # noqa: BLE001 — one bad frame must not kill the batch
            print(f"[{n}/{len(images)}] ERROR {path}: {e}")
            results.append({"image": path, "error": str(e)})
            continue
        rec = {"image": path, "matches": matches}
        results.append(rec)
        top = matches[0] if matches else None
        print("[%d/%d] %s → %d matches%s" % (
            n, len(images), Path(path).name, len(matches),
            ("  top=%s (%.4f) %.5f,%.5f L%gm" % (
                top["tile_key"], top["similarity"], top["coordinates"]["lat"],
                top["coordinates"]["lon"], top["level_m"]) if top else "")))
        if out_dir:
            (out_dir / (Path(path).stem + ".json")).write_text(
                json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")

    out = {"city": args.city, "k": args.k, "topk": args.topk, "device": loc.device,
           "n_images": len(images), "results": results}
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {len(results)} results → {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
