"""Fine reranking — vendored single-query version of geoLocalisation/patch_rerank/search_pyramid_s3.

Given the coarse (e2c supervlad) top-N candidate cell ids + the live query's DINO token grid, score
each candidate's pyramid against the query and return the top-K reranked cells.

Per cell: for every window position × pyramid level, mutual-NN(query tokens ↔ cell grid) → homography
inlier count / #query patches. Then:
  * pyramid (position) score = ``level_agg`` over its levels  (``sum`` = multi-scale evidence, the
    config that worked on kup: distR@250m top5 0.50; ``max`` = best level only),
  * cell score = ``cell_agg`` over its positions (``mean``),
  * predicted point = the best-scoring position, reported at its best level.

Verifier backend is env-switchable (SIAM_RERANK_VERIFY):
  * ``cv2`` (DEFAULT) — cv2 MAGSAC per (pos,level), pure CPU (run OUTSIDE the GPU lock by the caller).
  * ``torch`` — faithful GPU MAGSAC++ (vendored ``magsacpp_torch``), one BATCHED call over all
    (pos,level) crops of a cell. Must run UNDER the GPU lock (caller gates on the same env var).
The two share identical mutual-NN pairs (matched_coords) and identical level/cell aggregation — only
the inlier-count step differs. Numeric agreement with cv2 is close-but-not-bitwise (see
project_magsacpp_torch: median Δinliers=0, |Δ|≤1 ~53%); confirm distR before trusting a backend swap.
"""
from __future__ import annotations

import os

import numpy as np

from .matcher import _MIN_MATCHES, grid_keypoints, matched_coords, verify_inliers


def verify_backend() -> str:
    """'cv2' (default) or 'torch'. Read here AND by the siam caller (GPU-lock gating)."""
    return os.environ.get("SIAM_RERANK_VERIFY", "cv2").strip().lower()


def _mpp_config(reproj_thresh: float):
    """Build (MagsacppConfig, device, generator) from env. Light config that held kup quality:
    hyps=256, irls=1, closed_form, float32 (see project_magsacpp_torch)."""
    import torch

    from .magsacpp_torch import MagsacppConfig
    dev = os.environ.get("SIAM_RERANK_DEVICE", "cuda")
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    dtype = torch.float64 if os.environ.get("SIAM_MPP_DTYPE", "float32") == "float64" else torch.float32
    cfg = MagsacppConfig(
        sigma_max=float(os.environ.get("SIAM_MPP_SIGMA_MAX", "2.0")),
        inlier_threshold=float(reproj_thresh),
        max_hypotheses=int(os.environ.get("SIAM_MPP_HYPS", "256")),
        irls_iters=int(os.environ.get("SIAM_MPP_IRLS", "1")),
        minimal_solver=os.environ.get("SIAM_MPP_SOLVER", "closed_form"),
        dtype=dtype)
    gen = torch.Generator(device=dev).manual_seed(int(os.environ.get("SIAM_MPP_SEED", "0")))
    return cfg, dev, gen


def _torch_counts(mc, n_q, cfg, device, gen, minm):
    """Inlier counts for ALL (pos,level) crops of one cell in ONE batched GPU MAGSAC++ call.
    ``mc`` = [(qm, rm, n_mut)] in key order; crops with <min matches → 0. Mirrors the batched path
    in magsacpp_torch/evaluate (same source=rm→dest=qm convention)."""
    import torch

    from .magsacpp_torch import estimate_homography_magsacpp_batch
    counts = [0] * len(mc)
    idx = [j for j, (qm, rm, nm) in enumerate(mc) if n_q and nm >= minm]
    if not idx:
        return counts
    Nmax = max(mc[j][1].shape[0] for j in idx)
    B = len(idx)
    p1 = torch.zeros(B, Nmax, 2, dtype=cfg.dtype, device=device)   # source = rm (map)
    p2 = torch.zeros(B, Nmax, 2, dtype=cfg.dtype, device=device)   # dest   = qm (query)
    valid = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
    for b, j in enumerate(idx):
        qm, rm, _ = mc[j]
        m = rm.shape[0]
        p1[b, :m] = torch.as_tensor(np.asarray(rm, np.float64), dtype=cfg.dtype, device=device)
        p2[b, :m] = torch.as_tensor(np.asarray(qm, np.float64), dtype=cfg.dtype, device=device)
        valid[b, :m] = True
    cnts, _ = estimate_homography_magsacpp_batch(p1, p2, valid, config=cfg, generator=gen,
                                                 return_counts=True)
    cnts = cnts.tolist()
    for b, j in enumerate(idx):
        counts[j] = int(cnts[b])
    return counts


def _score_cell(cd, q_feat, q_xy, n_q: int, reproj_thresh: float, level_agg: str,
                verify: str = "cv2", cfg=None, device=None, gen=None):
    """→ (pyramid_score[n_pos], best_level[n_pos]) for one cell vs one query. Mutual-NN pairs are
    identical for both verifiers; only the inlier-count step (cv2 per-crop vs torch batched) differs."""
    minm = _MIN_MATCHES["homography"]
    keys = [(i, L) for i in range(cd.n_pos) for L in cd.levels]
    kp, mc = {}, []
    for (i, L) in keys:
        g = cd.grid(i, L)
        h, w, d = g.shape
        if (h, w) not in kp:
            kp[(h, w)] = grid_keypoints(h, w)
        mc.append(matched_coords(q_feat, g.reshape(h * w, d), q_xy, kp[(h, w)]))
    if verify == "torch":
        cnts = _torch_counts(mc, n_q, cfg, device, gen, minm)
        sc = {keys[j]: (cnts[j] / n_q if n_q else 0.0) for j in range(len(keys))}
    else:
        sc = {}
        for j, (i, L) in enumerate(keys):
            qm, rm, n_mut = mc[j]
            if n_q and n_mut >= minm:
                sc[(i, L)] = verify_inliers(qm, rm, "homography", "magsac", reproj_thresh).shape[0] / n_q
            else:
                sc[(i, L)] = 0.0
    pyr, blv = [], []
    for i in range(cd.n_pos):
        per = {L: sc[(i, L)] for L in cd.levels}
        bL = max(per, key=per.get)                               # reported level = best-scoring level
        pyr.append(sum(per.values()) if level_agg == "sum" else per[bL])
        blv.append(int(bL))
    return pyr, blv


def _aggregate(pyr, blv, cd, cell_agg: str):
    arr = np.asarray(pyr, float)
    cs = {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}[cell_agg]
    bi = int(arr.argmax())                                       # point = best position in the cell
    return cs, float(cd.lat[bi]), float(cd.lon[bi]), int(blv[bi])


def _prefetched_cells(store, cand_cell_ids, bufsize: int = 2):
    """Yield (cell_id, CellData) while a background thread reads the NEXT cells from
    disk — so the 400 MiB/cell read overlaps the caller's GPU work (the pipeline
    that turns serial read+compute into max(read, compute)). The reader touches
    only disk/CPU, never the GPU, so it is safe to run while the caller holds the
    GPU lock. cell → None on a read error (caller skips)."""
    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=bufsize)

    def _worker():
        for cid in cand_cell_ids:
            try:
                cd = store.cell(cid)
            except Exception:                                    # noqa: BLE001
                cd = None
            q.put((cid, cd))
        q.put(None)

    threading.Thread(target=_worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def _score_cell_multiframe(cd, q_norm_list, q_xys, reproj_thresh, level_agg,
                           cfg, device, gen, minm, mdtype):
    """Score ONE cell against ALL F query frames, GPU-batched over the cell grids.

    q_norm_list: list of F L2-normalised query token grids on `device` (fp16 on
    cuda); each is (n_q_f, D). n_q CAN DIFFER per frame (the e2c encoder returns a
    variable token count), so frames are LOOPED, not stacked. q_xys: the matching
    list of (n_q_f, 2) patch coords. The cell block is uploaded once; for each
    frame ONE batched matmul covers all (pos,level) grids; MAGSAC runs in ONE
    batched call over all F*(pos*level) crops. Returns a per-frame list of
    (cell_score, lat, lon, level_m) — the caller takes the best frame."""
    import torch

    F_ = len(q_norm_list)
    keys = [(i, L) for i in range(cd.n_pos) for L in cd.levels]
    G = len(keys)
    blk = np.asarray(cd._block)                                  # (P, L, H, W, D) fp16
    P, L, H, W, D = blk.shape
    N_r = H * W
    r_xy = grid_keypoints(H, W)
    block_t = torch.from_numpy(np.ascontiguousarray(blk).reshape(P * L, N_r, D)).to(device)
    block_t = torch.nn.functional.normalize(block_t.to(mdtype), dim=-1)      # (G, N_r, D)
    block_tT = block_t.transpose(-1, -2)                                     # (G, D, N_r)

    mc = [None] * (F_ * G)                                       # flat (frame, grid), ordered f*G+gi
    for f in range(F_):
        Qf = q_norm_list[f]                                      # (n_q, D) normalised
        sims = torch.matmul(Qf, block_tT)                        # (G, n_q, N_r) — batched over grids
        q2r = sims.argmax(2).cpu().numpy()                       # (G, n_q)
        r2q = sims.argmax(1).cpu().numpy()                       # (G, N_r)
        q_xy_f = q_xys[f]
        arq = np.arange(q2r.shape[1])
        for gi in range(G):
            q2r_g, r2q_g = q2r[gi], r2q[gi]
            mutual = r2q_g[q2r_g] == arq                         # mutual-NN on CPU (cheap)
            q_idx = arq[mutual]
            r_idx = q2r_g[mutual]
            mc[f * G + gi] = (q_xy_f[q_idx], r_xy[r_idx], int(q_idx.shape[0]))
    cnts = _torch_counts(mc, 1, cfg, device, gen, minm)          # flat F*G inlier counts (n_q=gate)

    out = []
    for f in range(F_):
        n_q = int(q_norm_list[f].shape[0])
        sc = {keys[gi]: (cnts[f * G + gi] / n_q if n_q else 0.0) for gi in range(G)}
        pyr, blv = [], []
        for i in range(cd.n_pos):
            per = {Lm: sc[(i, Lm)] for Lm in cd.levels}
            bL = max(per, key=per.get)
            pyr.append(sum(per.values()) if level_agg == "sum" else per[bL])
            blv.append(int(bL))
        arr = np.asarray(pyr, float)
        bi = int(arr.argmax())
        out.append((float(arr.mean()), float(cd.lat[bi]), float(cd.lon[bi]), int(blv[bi])))
    return out


def rerank_multiframe(q_feats, q_xys, cand_cell_ids, store, *, reproj_thresh: float = 2.0,
                      topk: int = 10, level_agg: str = "sum", cell_agg: str = "mean",
                      prefetch: bool = True) -> list[dict]:
    """Multi-frame fine rerank: score EVERY candidate cell against ALL F query
    frames (GPU-fp16 batched match + batched MAGSAC), keep each cell's BEST frame,
    return the top-`topk` DISTINCT cells.

    q_feats: list of F token grids (each (n_q_f, D) torch; n_q may differ per frame).
    q_xys:   list of F (n_q_f, 2) patch-coord arrays, aligned with q_feats.
    Returns [{cell_id, cell_score, lat, lon, level_m, frame_pos, rank}] sorted desc,
    length ≤ topk, one entry per cell. `frame_pos` indexes q_feats (the frame that
    scored this cell best). `cell_agg` kept for signature parity (mean, as the
    single-frame path). Matching is fp16 on cuda, fp32 on cpu."""
    import torch

    if not q_feats:
        return []
    cfg, device, gen = _mpp_config(reproj_thresh)
    mdtype = torch.float16 if str(device) != "cpu" else torch.float32
    q_norm_list = [torch.nn.functional.normalize(f.to(device).to(mdtype), dim=-1) for f in q_feats]
    q_xys = [np.asarray(x, np.float64) for x in q_xys]
    minm = _MIN_MATCHES["homography"]

    cand = [c for c in cand_cell_ids if store.has(c)]
    src = _prefetched_cells(store, cand) if prefetch else ((c, store.cell(c)) for c in cand)
    recs = []
    for cid, cd in src:
        if cd is None:
            continue
        per_frame = _score_cell_multiframe(cd, q_norm_list, q_xys, reproj_thresh, level_agg,
                                           cfg, device, gen, minm, mdtype)
        bf = int(np.argmax([p[0] for p in per_frame]))           # best frame for this cell
        cs, blat, blon, bL = per_frame[bf]
        recs.append({"cell_id": cid, "cell_score": cs, "lat": blat, "lon": blon,
                     "level_m": bL, "frame_pos": bf})
    recs.sort(key=lambda r: r["cell_score"], reverse=True)
    recs = recs[:topk]
    for rank, r in enumerate(recs, 1):
        r["rank"] = rank
    return recs


def rerank(q_feat, q_xy, cand_cell_ids, store, *, reproj_thresh: float = 2.0, topk: int = 5,
           level_agg: str = "sum", cell_agg: str = "mean") -> list[dict]:
    """Coarse top-N cell ids + query token grid → top-K reranked cells.

    q_feat: (N, D) torch tokens of the query frame (same DINO space as the store).
    q_xy:   (N, 2) numpy patch coords of those tokens.
    cand_cell_ids: coarse shortlist (e.g. top-100), cell ids matching the store manifest.
    Returns [{cell_id, cell_score, lat, lon, level_m}] sorted desc, length ≤ topk.

    Verifier chosen by SIAM_RERANK_VERIFY (cv2 default / torch GPU MAGSAC++).
    """
    n_q = int(q_feat.shape[0])
    verify = verify_backend()
    cfg = device = gen = None
    if verify == "torch":
        cfg, device, gen = _mpp_config(reproj_thresh)
    recs = []
    for cid in cand_cell_ids:
        if not store.has(cid):
            continue
        cd = store.cell(cid)
        pyr, blv = _score_cell(cd, q_feat, q_xy, n_q, reproj_thresh, level_agg, verify, cfg, device, gen)
        cs, blat, blon, bL = _aggregate(pyr, blv, cd, cell_agg)
        recs.append({"cell_id": cid, "cell_score": cs, "lat": blat, "lon": blon, "level_m": bL})
    recs.sort(key=lambda r: r["cell_score"], reverse=True)
    for rank, r in enumerate(recs[:topk], 1):
        r["rank"] = rank
    return recs[:topk]
