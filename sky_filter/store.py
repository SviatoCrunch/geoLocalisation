"""Mask artifact store: one compressed npz per frame + a manifest with a fingerprint.

The manifest (backend/version/shape/sha per frame) is the provenance record downstream
asserts against — so a run can prove it used the intended sky masks.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .schemas import SkyMask


def _safe(frame_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", frame_id)


class MaskStore:
    def __init__(self, root):
        self.root = Path(root).expanduser()
        self.masks_dir = self.root / "masks"
        self.manifest_path = self.root / "manifest.json"

    # ── read ──────────────────────────────────────────────────────────────────
    def _manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def load(self, frame_id: str) -> Optional[np.ndarray]:
        p = self.masks_dir / f"{_safe(frame_id)}.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return z["keep"].astype(bool)

    def has(self, frame_id: str) -> bool:
        return (self.masks_dir / f"{_safe(frame_id)}.npz").exists()

    # ── write ─────────────────────────────────────────────────────────────────
    def save(self, sm: SkyMask) -> None:
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        keep = np.asarray(sm.keep, bool)
        np.savez_compressed(self.masks_dir / f"{_safe(sm.frame_id)}.npz", keep=keep)
        man = self._manifest()
        man[sm.frame_id] = {
            "backend": sm.backend, "version": sm.version,
            "shape": list(keep.shape), "sky_fraction": round(float(1.0 - keep.mean()), 6),
            "sha": hashlib.sha256(np.ascontiguousarray(keep).tobytes()).hexdigest()[:16]}
        self.manifest_path.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")

    def fingerprint(self) -> str:
        return "sky:" + hashlib.sha256(
            json.dumps(self._manifest(), sort_keys=True).encode()).hexdigest()[:32]

    def frame_ids(self) -> list:
        return sorted(self._manifest())
