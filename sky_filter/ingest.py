"""Ingest GT sky masks (``*__Sky.png`` from ``GT_flat_mask``) into the precomputed store.

Rule (confirmed): **precomputed where a GT sky mask exists, neural where it doesn't.**
So we store a keep-mask ONLY for frames that have a ``__Sky.png``; frames without one are
simply absent from the store → the resolver falls through to the neural backend.

The GT mask marks SKY (non-zero = sky); the store keeps GROUND, so we INVERT:
``keep = ~(sky > 0)``. Frame id = ``"<city>:<stem>"`` where ``stem`` is the mask filename
without the ``__Sky.png`` suffix (matches the frame ids the rest of the pipeline uses).

The image reader is injectable (default = PIL) so this is testable without Pillow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import SkyMask


def _default_reader(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def ingest_sky_masks(mask_dir, city: str, store, *, suffix: str = "__Sky.png",
                     image_reader=None) -> dict:
    """Populate ``store`` (precomputed backend) from a directory of ``*__Sky.png`` masks.

    Returns a summary ``{ingested, skipped_bad, frame_ids, fingerprint}``. A frame whose
    stem can't form a valid id (e.g. the corrupt ``8.59…`` latitude) is skipped + logged.
    """
    reader = image_reader or _default_reader
    ingested, skipped_bad, frame_ids = 0, [], []
    for p in sorted(Path(mask_dir).expanduser().glob(f"*{suffix}")):
        stem = p.name[:-len(suffix)]
        if not stem or _looks_corrupt(stem):
            skipped_bad.append(stem)
            continue
        sky = np.asarray(reader(p)) > 0
        keep = ~sky
        store.save(SkyMask(frame_id=f"{city}:{stem}", keep=keep,
                           backend="precomputed", version="gt"))
        ingested += 1
        frame_ids.append(f"{city}:{stem}")
    return {"ingested": ingested, "skipped_bad": skipped_bad,
            "frame_ids": frame_ids, "fingerprint": store.fingerprint()}


def _looks_corrupt(stem: str) -> bool:
    """Flag the known bad pattern: a leading lat token like ``8.59…`` (missing the '4').

    Heuristic only for ``<lat>_<lon>`` names; UUID/other stems are always accepted.
    """
    head = stem.split("_", 1)[0]
    try:
        lat = float(head)
    except ValueError:
        return False                      # not a lat_lon name → accept as-is
    return not (44.0 <= lat <= 53.0)      # Ukraine lat band; 8.59 is out → corrupt
