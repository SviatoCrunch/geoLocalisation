"""Precompute pipeline: run the resolver over frames once, cache masks + manifest.

This is the FIRST stage of the drone pipeline — every downstream op reads the cached
keep-mask by frame_id (single source of truth), so sky filtering is consistent and never
recomputed.
"""
from __future__ import annotations

from collections import Counter


def precompute(frame_ids, image_loader, resolver, store, *, skip_existing: bool = True) -> dict:
    """For each frame: resolve a keep-mask (precomputed → neural) and store it.

    ``image_loader(frame_id) -> image`` is only needed by the neural fallback (may return
    None for precomputed-only). ``skip_existing`` avoids recomputing frames already stored.
    """
    by_backend, n_skipped = Counter(), 0
    for fid in frame_ids:
        if skip_existing and store.has(fid):
            n_skipped += 1
            continue
        image = image_loader(fid) if image_loader is not None else None
        sm = resolver.resolve(fid, image)
        store.save(sm)
        by_backend[sm.backend] += 1
    return {"n_frames": len(list(frame_ids)) if hasattr(frame_ids, "__len__") else None,
            "n_written": int(sum(by_backend.values())), "n_skipped": n_skipped,
            "by_backend": dict(by_backend), "fingerprint": store.fingerprint()}
