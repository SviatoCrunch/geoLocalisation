"""Consume an ALREADY-SPLIT frame tree (e.g. ``frames_good``) directly.

``frames_good`` is extracted frames already grouped by the content-adaptive
split into ``<station>/<date>/<chunk>/<sub_NN>/<chunk>_fNNN.jpg``. There is no
video to decode and no split to recompute — each ``sub_NN`` folder already IS a
sub-video. This module walks the tree and yields one group per chunk, so the
pipeline can go straight to quality -> diversity -> day/night.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")                       # S3 folder style DD.MM.YYYY
FRAME_RE = re.compile(r"^(?P<chunk>.+)_f(?P<idx>\d+)\.(?:jpg|jpeg|png)$", re.I)


@dataclass
class ChunkGroup:
    station: str
    date: str
    chunk: str
    chunk_dir: Path
    # sub_id -> sorted list of (frame_idx, Path)
    subs: dict[str, list[tuple[int, Path]]] = field(default_factory=dict)


def iter_chunk_groups(root) -> Iterator[ChunkGroup]:
    """Yield one :class:`ChunkGroup` per (station, date, chunk) found under ``root``.

    Sub-video id is the immediate parent folder name (``sub_NN``); if a chunk's
    frames are not nested in a ``sub_*`` folder they collapse to a single
    ``sub_01``.
    """
    root = Path(root)
    groups: dict[tuple[str, str, str], ChunkGroup] = {}
    members: dict[tuple[str, str, str], dict[str, list[tuple[int, Path]]]] = defaultdict(
        lambda: defaultdict(list))

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if not m:
            continue
        date_dir = next((a for a in p.parents if DATE_RE.match(a.name)), None)
        if date_dir is None:
            continue
        station = date_dir.parent.name
        chunk = m.group("chunk")
        key = (station, date_dir.name, chunk)
        sub = p.parent.name if p.parent.name.lower().startswith("sub") else "sub_01"
        chunk_dir = date_dir / chunk
        if key not in groups:
            groups[key] = ChunkGroup(station=station, date=date_dir.name, chunk=chunk,
                                     chunk_dir=chunk_dir)
        members[key][sub].append((int(m.group("idx")), p))

    for key in sorted(groups):
        g = groups[key]
        g.subs = {sub: sorted(members[key][sub]) for sub in sorted(members[key])}
        yield g
