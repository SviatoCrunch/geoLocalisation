"""Thin adapters over existing project code + shared helpers (no core changes).

Everything the diagnostics need is built here by CALLING the real project functions, so the
audits stay consistent with training/eval:
  * model      — rebuilt from the checkpoint's ``resolved_config`` via ``siam_e2c_model.build_e2c_model``
  * split      — ``geo_train_batching.adapters.split_relevance.build_split_relevance``
  * loaders    — ``geo_e2c_train.data.TileGridLoader`` / ``QueryTokenStore``
  * gallery V  — ``geo_e2c_train.eval.build_gallery_V`` (rebuilt from the loaded checkpoint)
  * scoring    — ``model.score`` / ``model.score_with_details``
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SCHEMA_VERSION = 1
_COS = 0.657                                  # EPSG:3857 → true metres at ~48.9° (cities' latitude)
DIST_THR = (250.0, 500.0, 1000.0)


# ── IO ────────────────────────────────────────────────────────────────────────────
def write_json(path, obj) -> None:
    obj = {"schema_version": SCHEMA_VERSION, **obj}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_jsonable), encoding="utf-8")


def _jsonable(x):
    import numpy as _np
    if isinstance(x, (_np.floating,)):
        return float(x)
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, _np.ndarray):
        return x.tolist()
    return str(x)


# ── model / checkpoint ──────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path):
    import torch
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def cfg_from_resolved(resolved: dict, assign_override=None):
    from dataclasses import fields
    from siam_e2c_model.config import E2cModelConfig
    keep = {f.name for f in fields(E2cModelConfig)}
    kw = {k: resolved[k] for k in resolved if k in keep}
    if assign_override:
        kw["assign_path"] = assign_override
    return E2cModelConfig(**kw)


def load_model(ckpt, device="cpu", assign_override=None, strict=True):
    """Rebuild from resolved_config + load weights. Returns (model, info)."""
    from siam_e2c_model.model import build_e2c_model
    resolved = ckpt.get("resolved_config", {})
    cfg = cfg_from_resolved(resolved, assign_override)
    model = build_e2c_model(cfg, device=device)
    missing, unexpected = [], []
    try:
        ret = model.load_state_dict(ckpt["model"], strict=strict)
        missing = list(getattr(ret, "missing_keys", []) or [])
        unexpected = list(getattr(ret, "unexpected_keys", []) or [])
    except Exception as e:                    # surface, don't hide (audit reports it)
        raise
    model.eval()
    return model, {"epoch": ckpt.get("epoch"), "pyramid_mode": model.pyramid_mode,
                   "agg": cfg.agg, "k": cfg.k, "d_token": cfg.d_token,
                   "missing_keys": missing, "unexpected_keys": unexpected,
                   "resolved_config": resolved}


def param_fingerprints(model) -> dict:
    """sha256 of trainable params, split into map-branch (affects gallery V) vs query/gate."""
    import torch
    core = model.core
    groups = {"map_branch": [], "query_branch": [], "gate_temp": [], "all_trainable": []}
    named = dict(core.named_parameters())
    for name, p in named.items():
        if not p.requires_grad:
            continue
        groups["all_trainable"].append((name, p))
        if name.startswith(("group_proj", "map_head")):
            groups["map_branch"].append((name, p))
        elif name.startswith("drone_head"):
            groups["query_branch"].append((name, p))
        elif name.startswith(("rho", "scale_gate")):
            groups["gate_temp"].append((name, p))
    out = {}
    for g, items in groups.items():
        h = hashlib.sha256()
        for name, p in sorted(items):
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        out[g] = h.hexdigest()[:32] if items else None
    return out


# ── data / split ────────────────────────────────────────────────────────────────────
def parse_kv(items):
    out = {}
    for a in items:
        c, p = a.split("=", 1)
        out[c] = p
    return out


@dataclass
class DiagContext:
    args: object
    model: object
    model_info: dict
    device: str
    tiles: object                 # non-caching eval TileGridLoader
    store: object                 # QueryTokenStore
    sr: dict = field(default_factory=dict)     # which -> SplitRelevance
    _galV: object = None

    def galV(self):
        """Full-gallery V from the loaded checkpoint (built once, shared)."""
        if self._galV is None:
            from geo_e2c_train.eval import build_gallery_V
            any_sr = self.sr["train"]
            self._galV = build_gallery_V(self.model, self.tiles, any_sr.tile_ids,
                                         self.args.eval_chunk, self.device, progress=True)
        return self._galV


def build_context(args) -> DiagContext:
    from geo_train_batching.adapters.split_relevance import build_split_relevance
    from geo_e2c_train.data import TileGridLoader, QueryTokenStore

    ckpt = load_checkpoint(args.checkpoint)
    model, info = load_model(ckpt, device=args.device, assign_override=args.assign,
                             strict=not args.allow_nonstrict)
    galleries = parse_kv(args.galleries)
    queries = parse_kv(args.queries)
    tiles = TileGridLoader(galleries, grid_size=None, cache=False)
    store = QueryTokenStore(queries)
    which = ["train", "val"] + (["test"] if args.include_test else [])
    sr = {w: build_split_relevance(args.split_config, args.split_json, w,
                                   query_size_m=args.query_size_m, safe_eps_area=args.safe_eps_area)
          for w in which}
    return DiagContext(args=args, model=model, model_info=info, device=args.device,
                       tiles=tiles, store=store, sr=sr)


# ── scoring / ranking helpers (reuse model.score; return per-query detail) ─────────────
def q_embed(model, store, qid, dev):
    import torch
    return model.encode_query(store.tokens(qid).to(dev))


@torch.no_grad()
def full_gallery_per_query(model, store, sr, galV, dev, tile_chunk, ks=(1, 5, 10, 20)):
    """Per-query ranks + distances vs the full gallery. Returns (rows, aggregate)."""
    qids = [q for q in sr.query_ids if store.has(q)]
    qid_row = {q: i for i, q in enumerate(sr.query_ids)}
    rows = []
    if not qids:
        return rows, {"n": 0, "median_rank": float("nan"),
                      **{f"R@{k}": 0.0 for k in ks}, "hits": {f"R@{k}": 0 for k in ks}}
    Q = torch.stack([q_embed(model, store, q, dev) for q in qids])
    S = model.score(Q, galV, tile_chunk=tile_chunk)
    order = S.argsort(dim=1, descending=True).cpu().numpy()
    ranks, hit, dists = [], {k: 0 for k in ks}, []
    for bi, q in enumerate(qids):
        pos = set(int(x) for x in sr.relevance.pos_of(qid_row[q]))
        if not pos:
            continue
        ranked = order[bi]
        r = next((j for j, t in enumerate(ranked) if int(t) in pos), len(ranked))
        top1 = int(ranked[0])
        d = float(np.linalg.norm(sr.tile_xy[top1] - sr.q_xy[qid_row[q]])) * _COS
        ranks.append(r + 1); dists.append(d)
        for k in ks:
            if r < k:
                hit[k] += 1
        rows.append({"query_id": q, "exact_rank": r + 1, "n_positives": len(pos),
                     "top1_tile_row": top1, "top1_dist_m": d})
    n = len(ranks)
    dm = np.array(dists) if dists else np.array([np.nan])
    agg = {"n": n, "median_rank": float(np.median(ranks)) if ranks else float("nan"),
           **{f"R@{k}": (hit[k] / n if n else 0.0) for k in ks},
           "hits": {f"R@{k}": int(hit[k]) for k in ks},
           "median_dist_m": float(np.median(dm)),
           **{f"distR@{int(t)}m": float((dm <= t).mean()) for t in DIST_THR}}
    return rows, agg


def rank_matrix(S_np, qids, qid_row, sr, ks=(1, 5, 10, 20)):
    """Recall@ks + median rank from a precomputed score matrix (B,M) aligned to ``qids``."""
    order = np.argsort(-S_np, axis=1)
    ranks, hit = [], {k: 0 for k in ks}
    for bi, q in enumerate(qids):
        pos = set(int(x) for x in sr.relevance.pos_of(qid_row[q]))
        if not pos:
            continue
        r = next((j for j, t in enumerate(order[bi]) if int(t) in pos), order.shape[1])
        ranks.append(r + 1)
        for k in ks:
            if r < k:
                hit[k] += 1
    n = len(ranks)
    return {"n": n, "median_rank": float(np.median(ranks)) if ranks else float("nan"),
            **{f"R@{k}": (hit[k] / n if n else 0.0) for k in ks},
            "hits": {f"R@{k}": int(hit[k]) for k in ks}}


@torch.no_grad()
def candidate_set_metrics(model, store, sr, tiles, dev, tile_chunk, ks=(1, 5, 10)):
    """A) Metric on the TRAIN candidate set actually used by the loss: for each query, rank of its
    positive among its DSS candidate set (pos ∪ safe from the relevance table). NOT the full gallery,
    NOT B×B in-batch — the per-query candidate pool the objective optimises."""
    qids = [q for q in sr.query_ids if store.has(q)]
    qid_row = {q: i for i, q in enumerate(sr.query_ids)}
    ranks, hit, cand_counts = [], {k: 0 for k in ks}, []
    for q in qids:
        i = qid_row[q]
        pos = sorted(int(x) for x in sr.relevance.pos_of(i))
        safe = sorted(int(x) for x in sr.relevance.safe_of(i))
        cand = pos + [s for s in safe if s not in set(pos)]
        if not pos or len(cand) < 2:
            continue
        G = tiles.stack([sr.tile_ids[r] for r in cand]).float().to(dev)
        V = model.build_V(G)
        Q = q_embed(model, store, q, dev).unsqueeze(0)
        s = model.score(Q, V)[0].detach().cpu().numpy()          # (len(cand),)
        order = np.argsort(-s)
        pos_local = set(range(len(pos)))                         # positives are the first entries
        r = next((j for j, c in enumerate(order) if c in pos_local), len(order))
        ranks.append(r + 1); cand_counts.append(len(cand))
        for k in ks:
            if r < k:
                hit[k] += 1
    n = len(ranks)
    return {"n_queries": n, "candidate_count_median": int(np.median(cand_counts)) if cand_counts else 0,
            "median_rank": float(np.median(ranks)) if ranks else float("nan"),
            **{f"R@{k}": (hit[k] / n if n else 0.0) for k in ks},
            "hits": {f"R@{k}": int(hit[k]) for k in ks}}
