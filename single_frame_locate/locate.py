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
import math
import os
import re
import shutil
import sys
import time
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


# ── metrics + timing helpers ────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


_LAT_RANGE, _LON_RANGE = (44.0, 53.0), (20.0, 45.0)      # Ukraine AO sanity window


def _gt_latlon_from_name(path: str):
    """GT lat/lon from a ``<id>_<lat>_<lon>.jpg`` filename (the GT_flat convention) — the first
    adjacent float pair that falls in the AO lat/lon window. None when no plausible pair is found."""
    floats = [float(t) for t in re.findall(r"-?\d+\.\d+", Path(path).stem)]
    for i in range(len(floats) - 1):
        lat, lon = floats[i], floats[i + 1]
        if _LAT_RANGE[0] <= lat <= _LAT_RANGE[1] and _LON_RANGE[0] <= lon <= _LON_RANGE[1]:
            return lat, lon
    return None


_DISTS = (250, 500, 1000)                                 # recall distance bins (m)
_KS = (1, 5, 10)                                          # top-K bins


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _summarize(results, topk: int) -> dict:
    """Aggregate timing + localization metrics over the per-frame results."""
    done = [r for r in results if "matches" in r]
    secs = [r["seconds"] for r in results if "seconds" in r]
    gted = [r for r in done if r.get("gt") and r["matches"]]
    ks = [k for k in _KS if k <= topk] or [topk]
    s = {"n_images": len(results), "n_localized": len(done), "n_with_gt": len(gted)}
    if secs:
        s["timing_s"] = {"total": round(sum(secs), 1), "mean": round(sum(secs) / len(secs), 2),
                         "median": round(_median(secs), 2), "min": round(min(secs), 2),
                         "max": round(max(secs), 2)}
    if gted:
        top1 = [r["top1_err_m"] for r in gted if r.get("top1_err_m") is not None]
        s["top1_err_m"] = {"median": round(_median(top1), 1), "mean": round(sum(top1) / len(top1), 1),
                           "min": round(min(top1), 1), "max": round(max(top1), 1)}
        recall = {}
        for k in ks:
            for d in _DISTS:
                hits = sum(1 for r in gted
                           if any(m.get("dist_m_to_gt", 9e18) <= d for m in r["matches"][:k]))
                recall["top%d@%dm" % (k, d)] = round(hits / len(gted), 4)
        s["recall"] = recall
    return s


def _print_summary(s: dict) -> None:
    print("\n==== SUMMARY ====")
    print("images=%d  localized=%d  with_gt=%d" % (s["n_images"], s["n_localized"], s["n_with_gt"]))
    t = s.get("timing_s")
    if t:
        print("time/frame s: mean=%.2f median=%.2f min=%.2f max=%.2f | total=%.1fs wall=%.1fs"
              % (t["mean"], t["median"], t["min"], t["max"], t["total"], s.get("wall_s", t["total"])))
    e = s.get("top1_err_m")
    if e:
        print("top1 err m: median=%.1f mean=%.1f min=%.1f max=%.1f"
              % (e["median"], e["mean"], e["min"], e["max"]))
    r = s.get("recall")
    if r:
        print("recall (fraction with a top-K cell within D of GT):")
        for k in _KS:
            row = ["top%d@%dm=%.3f" % (k, d, r["top%d@%dm" % (k, d)])
                   for d in _DISTS if "top%d@%dm" % (k, d) in r]
            if row:
                print("  " + "  ".join(row))


# ── GPU / load monitoring ───────────────────────────────────────────────────

def _vram_load():
    """(gpu_alloc_GB, gpu_reserved_GB, load1) — cheap per-frame GPU + host-load probe."""
    a = r = 0.0
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        r = torch.cuda.memory_reserved() / 1e9
    try:
        l1 = os.getloadavg()[0]
    except (OSError, AttributeError):
        l1 = -1.0
    return a, r, l1


# ── per-cell store → packed-block view (run the server fp16 fine path on per-cell data) ──

class _BlockCell:
    """Wrap a per-cell CellData so it exposes a packed ``(P, L, H, W, D)`` fp16 ``_block`` — the only
    extra thing ``rerank_multiframe`` needs. Lets the per-cell (S3) store run the EXACT server fp16
    fine path (fp16 batched match + batched GPU MAGSAC++) over identical token data."""

    def __init__(self, cd):
        self.cell_id = cd.cell_id
        self.lat, self.lon = cd.lat, cd.lon
        self.levels, self.n_pos = cd.levels, cd.n_pos
        h, w, d = cd.grid(0, self.levels[0]).shape
        blk = np.empty((self.n_pos, len(self.levels), h, w, d), np.float16)
        for i in range(self.n_pos):
            for j, lv in enumerate(self.levels):
                blk[i, j] = cd.grid(i, lv).numpy().astype(np.float16)
        self._block = blk


class _PackedLikeStore:
    """Per-cell store → packed-block view so ``rerank_multiframe`` runs over it unchanged."""

    def __init__(self, base):
        self._base = base
        self.cells = base.cells

    def has(self, cid):
        return self._base.has(cid)

    def cell(self, cid):
        return _BlockCell(self._base.cell(cid))


class SingleFrameLocator:
    """Lazy load of the e2c coarse model + gallery V + packed pyramid store; localize one frame.

    Mirrors SiamEngine: owns the gallery cache (V, keys, lat/lon/win), the coarse → fine wiring, and
    the reranked-cell → match-dict mapping. No AOI filter / pair render / GPU lock (single-process
    offline tool), so the GPU compute is the only thing that runs — exactly as production."""

    def __init__(self, ckpt: str, map_h5: str, assign_path: str, *, store: str,
                 device: str = "auto", city: str = "kram", rerank_k: int = 100,
                 rerank_topk: int = 10, rerank_reproj: float = 2.0, chunk: int = 256,
                 index_cache: str | None = None, rebuild: bool = False,
                 store_format: str = "packed", cache_dir: str = "/tmp/cellcache",
                 cache_cap: int = 256, match_fp16: bool = True):
        self.device = _resolve_device(device)
        # the batched GPU MAGSAC++ reads its device from this env var (reranker._mpp_config);
        # pin it to the same device so coarse + fine run together on the GPU, as on the server.
        os.environ.setdefault("SIAM_RERANK_DEVICE", str(self.device))
        # per-cell path verifier = GPU MAGSAC++ (the packed path always uses torch internally);
        # keep the two paths numerically equivalent for a single frame.
        os.environ.setdefault("SIAM_RERANK_VERIFY", "torch")

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
        self.cache_dir = cache_dir
        self.cache_cap = int(cache_cap)
        self.match_fp16 = bool(match_fp16)
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
            else:                                            # per-cell H5 store (local dir or s3://…/_index.json)
                from .rerank import CellStore
                self._store = CellStore(self.store_path, cache_dir=self.cache_dir,
                                        cache_cap=self.cache_cap)
        return self._store

    # -- coarse: one DINO forward → top-K candidate cells + the query token grid --
    @torch.no_grad()
    def _topk_full(self, rgb: np.ndarray, top: int):
        q, tokens, xy = self.backend.encode_query_full(rgb)
        sims = self.backend.score(q, self.V, self.chunk)
        order = np.argsort(-sims)[:top]
        return [(int(i), float(sims[i])) for i in order], tokens, xy

    # -- fine: GPU MAGSAC++ rerank of the candidate cells --
    def _rerank(self, tokens, xy, cand_tis: list[int], topk: int) -> list[dict]:
        store = self._load_store()
        cell_ids = [f"{self.city}:{self.keys[ti]}" for ti in cand_tis]
        if self.store_format == "packed" or self.match_fp16:
            # SERVER-EXACT fine path: fp16 batched match + batched GPU MAGSAC++ (rerank_multiframe, F=1).
            # Packed store feeds its /features block directly; a per-cell store is wrapped so each cell
            # exposes a packed fp16 _block (same token data) — identical fp16 compute to the server.
            from .rerank import rerank_multiframe
            s = store if self.store_format == "packed" else _PackedLikeStore(store)
            return rerank_multiframe([tokens.detach()], [np.asarray(xy, np.float64)], cell_ids, s,
                                     reproj_thresh=self.rerank_reproj, topk=topk,
                                     level_agg="sum", cell_agg="mean")
        # fp32 per-cell fallback (--no-fp16-match): single-frame rerank() with the torch verifier —
        # same MAGSAC math, but the mutual-NN match is fp32 (matched_coords upcasts). q_feat on CPU.
        from .rerank import rerank as rerank_single
        return rerank_single(tokens.detach().to("cpu").float(), np.asarray(xy, np.float64), cell_ids,
                             store, reproj_thresh=self.rerank_reproj, topk=topk,
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

    # -- fine rerank for a GROUP of frames: each cell of the UNION is read/built ONCE and matched
    #    against ALL frames in one batched fp16 pass (server _score_cell_multiframe compute); every
    #    frame then ranks ONLY its own coarse candidates → per-frame top-K, shared compute --
    def _rerank_group(self, tokens_list, xy_list, cand_tis_list, topk: int) -> list[list[dict]]:
        import torch

        from .rerank.reranker import (_MIN_MATCHES, _mpp_config, _prefetched_cells,
                                       _score_cell_multiframe)
        base = self._load_store()
        store = base if self.store_format == "packed" else _PackedLikeStore(base)
        per_frame_ids = [[f"{self.city}:{self.keys[ti]}" for ti in tis] for tis in cand_tis_list]
        union = list(dict.fromkeys(cid for ids in per_frame_ids for cid in ids if store.has(cid)))

        cfg, device, gen = _mpp_config(self.rerank_reproj)
        mdtype = torch.float16 if str(device) != "cpu" else torch.float32
        q_norm = [torch.nn.functional.normalize(t.detach().to(device).to(mdtype), dim=-1)
                  for t in tokens_list]
        q_xys = [np.asarray(x, np.float64) for x in xy_list]
        minm = _MIN_MATCHES["homography"]

        scored: dict = {}                                        # cell_id -> per-frame [(cs,lat,lon,lvl)]
        for cid, cd in _prefetched_cells(store, union):
            if cd is None:
                continue
            scored[cid] = _score_cell_multiframe(cd, q_norm, q_xys, self.rerank_reproj, "sum",
                                                 cfg, device, gen, minm, mdtype)
        out = []
        for f in range(len(tokens_list)):
            recs = []
            for cid in dict.fromkeys(per_frame_ids[f]):          # this frame's OWN candidates
                if cid in scored:
                    cs, lat, lon, lvl = scored[cid][f]
                    recs.append({"cell_id": cid, "cell_score": cs, "lat": lat, "lon": lon,
                                 "level_m": lvl})
            recs.sort(key=lambda r: r["cell_score"], reverse=True)
            recs = recs[:topk]
            for rank, r in enumerate(recs, 1):
                r["rank"] = rank
            out.append(recs)
        return out

    def localize_group(self, frames_bgr: list, *, top: int | None = None) -> list[list[dict]]:
        """Localize G frames together (shared cell reads + batched GPU match). Returns a per-frame
        list of top-K match dicts — each frame keeps its OWN answer. For G==1 this equals localize()."""
        top = int(top or self.rerank_topk or 10)
        tokens_list, xy_list, cand_list = [], [], []
        for img in frames_bgr:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            coarse, tokens, xy = self._topk_full(rgb, self.rerank_k)
            tokens_list.append(tokens)
            xy_list.append(xy)
            cand_list.append([ti for ti, _ in coarse])
        recs_per_frame = self._rerank_group(tokens_list, xy_list, cand_list, self.rerank_topk or top)
        return [[self._match_reranked(rec, r) for r, rec in enumerate(recs, 1)]
                for recs in recs_per_frame]


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
    ap.add_argument("--store-format", default="packed", choices=["packed", "percell"],
                    help="packed = /features GPU block (exact server path); percell = per-cell H5 "
                         "store (local dir or s3://…/_index.json), single-frame rerank, same GPU math")
    ap.add_argument("--cache-dir", default="/tmp/cellcache",
                    help="percell-S3 only: local LRU cache dir for downloaded cells")
    ap.add_argument("--cache-cap", type=int, default=256,
                    help="percell-S3 only: max cells kept in the local cache (bounds disk)")
    ap.add_argument("--no-fp16-match", action="store_true",
                    help="use fp32 mutual-NN matching (default = fp16, exactly like the server)")
    ap.add_argument("--empty-cache-every", type=int, default=25,
                    help="torch.cuda.empty_cache() every N frames (0=never) — curbs VRAM fragmentation")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="abort the batch if free disk on the cache/root drops below this (anti-hang guard)")
    ap.add_argument("--frame-batch", type=int, default=1,
                    help="localize G frames per group (shared cell reads + batched GPU match) for speed; "
                         "each frame still keeps its own top-K. 1 = one frame at a time.")
    ap.add_argument("--index-cache", default=None, help="optional path to cache/reuse the gallery V")
    ap.add_argument("--rebuild", action="store_true", help="force-rebuild the gallery V cache")
    ap.add_argument("--out", default=None,
                    help="write the combined JSON (list of per-image results) here (default: stdout)")
    ap.add_argument("--out-dir", default=None,
                    help="also write one <stem>.json per image into this dir")
    ap.add_argument("--no-gt", action="store_true",
                    help="do NOT parse GT lat/lon from filenames (skip accuracy metrics)")
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
                             store_format=args.store_format, cache_dir=args.cache_dir,
                             cache_cap=args.cache_cap, match_fp16=not args.no_fp16_match)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # tqdm time log — falls back to plain prints if tqdm is absent
    try:
        from tqdm import tqdm
        bar = tqdm(total=len(images), unit="frame", dynamic_ncols=True)
        emit = tqdm.write
    except Exception:                                    # noqa: BLE001
        bar, emit = None, print

    def _emit(msg):
        (emit if bar else print)(msg)

    guard_path = loc.cache_dir if loc.store_format != "packed" else "/"
    g_size = max(1, args.frame_batch)
    results = []
    done = 0
    t_start = time.perf_counter()
    for c0 in range(0, len(images), g_size):
        chunk = images[c0:c0 + g_size]
        # anti-hang guard: stop cleanly before the disk fills (what hung the box before)
        if args.min_free_gb > 0:
            try:
                free_gb = shutil.disk_usage(guard_path).free / 1e9
            except OSError:
                free_gb = 1e9
            if free_gb < args.min_free_gb:
                _emit("ABORT: free disk %.1fGB < --min-free-gb %.1f" % (free_gb, args.min_free_gb))
                break
        imgs, paths_ok = [], []
        for p in chunk:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                _emit("SKIP unreadable %s" % p)
                results.append({"image": p, "error": "unreadable"})
            else:
                imgs.append(im)
                paths_ok.append(p)
        if not imgs:
            if bar:
                bar.update(len(chunk))
            continue
        t0 = time.perf_counter()
        try:
            per_frame = loc.localize_group(imgs) if g_size > 1 else [loc.localize(imgs[0])]
        except Exception as e:                           # noqa: BLE001 — one bad group must not kill the batch
            _emit("ERROR group @%d (%d frames): %s" % (c0, len(imgs), e))
            for p in paths_ok:
                results.append({"image": p, "error": str(e)})
            if bar:
                bar.update(len(chunk))
            continue
        grp_s = time.perf_counter() - t0
        per_s = grp_s / len(imgs)
        ga, gr, l1 = _vram_load()
        for p, matches in zip(paths_ok, per_frame):
            done += 1
            gt = None if args.no_gt else _gt_latlon_from_name(p)
            rec = {"image": p, "seconds": round(per_s, 3), "group_seconds": round(grp_s, 3),
                   "group_size": len(imgs), "gpu_alloc_gb": round(ga, 2),
                   "gpu_reserved_gb": round(gr, 2), "load1": round(l1, 2), "matches": matches}
            if gt is not None:
                rec["gt"] = {"lat": gt[0], "lon": gt[1]}
                for m in matches:
                    m["dist_m_to_gt"] = round(_haversine_m(gt[0], gt[1], m["coordinates"]["lat"],
                                                           m["coordinates"]["lon"]), 1)
                rec["top1_err_m"] = matches[0]["dist_m_to_gt"] if matches else None
            results.append(rec)
            top = matches[0] if matches else None
            _emit("[%d/%d] %s  %.1fs/f%s%s" % (
                done, len(images), Path(p).name, per_s,
                ("  top=%s (%.4f) %.5f,%.5f L%gm" % (
                    top["tile_key"], top["similarity"], top["coordinates"]["lat"],
                    top["coordinates"]["lon"], top["level_m"]) if top else "  (no matches)"),
                ("  err=%.0fm" % rec["top1_err_m"]) if rec.get("top1_err_m") is not None else ""))
            if out_dir:
                (out_dir / (Path(p).stem + ".json")).write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        if bar:
            bar.update(len(chunk))
            bar.set_postfix_str("G=%d %.1fs/f vram=%.1fGB load=%.1f" % (len(imgs), per_s, gr, l1))
        if args.empty_cache_every and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if bar:
        bar.close()

    summary = _summarize(results, args.topk)
    summary["wall_s"] = round(time.perf_counter() - t_start, 1)
    out = {"city": args.city, "k": args.k, "topk": args.topk, "device": loc.device,
           "n_images": len(images), "summary": summary, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {len(results)} results → {args.out}")
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
