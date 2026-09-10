"""Standalone ``load_assign_weight`` — vendored verbatim from
``siam_model_stage3/model/factory.py`` (depends only on torch). Resolves a frozen
SuperVLAD assignment weight (K, D) from a recast blob so ``siam_e2c_model`` needs no
external repo.
"""
from __future__ import annotations


def load_assign_weight(path, *, k=None, d=None):
    """Robustly resolve a frozen SuperVLAD assignment weight (K,D) from a file.

    Accepts: a raw (K,D) tensor; a dict with ``assign_init_weight``/``assign_weight``; a
    state_dict with ``agg.assign.weight``/``assign.weight``; or ``centroids`` (+optional
    ``alpha``) → γ·Ĉ. Fails CLOSED (lists what it found + the expected (K,D)).
    """
    import torch
    import torch.nn.functional as F
    blob = torch.load(path, map_location="cpu", weights_only=False)
    w = None
    if torch.is_tensor(blob):
        w = blob
    elif isinstance(blob, dict):
        for key in ("assign_init_weight", "assign_weight"):
            if torch.is_tensor(blob.get(key)):
                w = blob[key]; break
        if w is None:
            sd = blob.get("state_dict", blob.get("model_state", blob))
            if isinstance(sd, dict):
                for key in ("agg.assign.weight", "assign.weight"):
                    if torch.is_tensor(sd.get(key)):
                        w = sd[key]; break
        if w is None and torch.is_tensor(blob.get("centroids")):
            C = F.normalize(blob["centroids"].float(), dim=1)
            w = float(blob.get("alpha", 1.0)) * C
    if w is None:
        found = list(blob.keys()) if isinstance(blob, dict) else type(blob).__name__
        raise ValueError(f"assign-init {path}: no assignment weight found. Found: {found}. "
                         "Expected a (K,D) tensor, or a key in {assign_init_weight, assign_weight, "
                         "agg.assign.weight, assign.weight}, or a centroids(+alpha) blob.")
    w = w.float()
    if w.dim() != 2:
        raise ValueError(f"assign-init: assignment must be 2-D (K,D); got {tuple(w.shape)}.")
    if (k and w.shape[0] != k) or (d and w.shape[1] != d):
        raise ValueError(f"assign-init: shape {tuple(w.shape)} != expected ({k},{d}); K/D must "
                         "match the dictionary (n_groups, token_dim).")
    return w
