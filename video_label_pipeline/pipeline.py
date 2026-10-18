"""Orchestrator: videos -> detailed annotation manifest (JSON).

Per video: decode frames -> content-adaptive split into sub-videos -> score
every frame's quality -> among the ``good`` frames pick ``k`` by maximum visual
diversity -> tag each picked frame day/night -> emit a detailed record.

``process_decoded`` is the pure core (frames already decoded, backend injected)
so it is unit-testable without OpenCV video decoding, torch, or S3.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .daynight import THRESHOLD, day_night
from .diversity import farthest_point_select
from .frames_tree import iter_chunk_groups
from .segment import boundary_context, segment_frames, to_feature
from .video_io import read_all_frames

_QUALITY_CLASSES = ("good", "usable", "bad")


@dataclass
class SplitRecipe:
    """Content-adaptive recipe. Defaults == the recipe that built ``frames_good``."""
    adaptive_thr: float = 3.0
    min_content: float = 15.0
    content_window: int = 2
    content_reduce: int = 4
    min_seg: int = 2
    width: int = 480
    backend: str = "content"
    tool: str = "subvideo_homography"

    def as_dict(self) -> dict:
        return asdict(self)


def _select_subvideo(positions, frames, backend, k, thr):
    """Build one sub-video record (quality counts + k diverse good selections)."""
    sub_bgr = [frames[p][2] for p in positions]
    labels = backend.score(sub_bgr)

    counts = {c: 0 for c in _QUALITY_CLASSES}
    good = []                                  # (position, confidence)
    for p, (lab, conf) in zip(positions, labels):
        counts[lab] = counts.get(lab, 0) + 1
        if lab == "good":
            good.append((p, conf))

    selected = []
    if good:
        embeds = backend.embed([frames[p][2] for p, _ in good])
        for rank, gi in enumerate(farthest_point_select(embeds, k), start=1):
            pos, conf = good[gi]
            fidx, ts, bgr = frames[pos]
            dn, cval = day_night(bgr, thr)
            selected.append({
                "rank": rank, "frame_idx": fidx, "timestamp_s": round(ts, 3),
                "label": "good", "confidence": round(float(conf), 4),
                "day_night": dn, "chroma": round(cval, 2) if cval is not None else None,
            })
        selected.sort(key=lambda r: r["rank"])
    return counts, selected


def process_decoded(ref, fps, n_frames, frames, backend, recipe: SplitRecipe,
                    k: int = 5, thr: float = THRESHOLD) -> dict:
    """Pure core. ``frames`` is a list of ``(frame_idx, timestamp_s, bgr)``."""
    feats = [to_feature(f[2], recipe.width, recipe.content_reduce) for f in frames]
    segs, deltas, _ = segment_frames(feats, recipe.adaptive_thr, recipe.min_content,
                                     recipe.content_window, recipe.min_seg)

    subvideos = []
    for si, positions in enumerate(segs):
        start_pos, end_pos = positions[0], positions[-1]
        start_s, end_s = frames[start_pos][1], frames[end_pos][1]
        counts, selected = _select_subvideo(positions, frames, backend, k, thr)
        boundary = None
        if si > 0:                             # first sub starts the clip, no cut
            boundary = {"cut_before_frame": frames[start_pos][0],
                        **boundary_context(deltas, start_pos, recipe.content_window,
                                           recipe.min_content, recipe.adaptive_thr)}
        subvideos.append({
            "sub_id": f"sub_{si + 1:02d}",
            "start_frame": frames[start_pos][0],
            "end_frame": frames[end_pos][0],
            "n_frames": len(positions),
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration_s": round(end_s - start_s, 3),
            "boundary": boundary,
            "quality": {c: counts.get(c, 0) for c in _QUALITY_CLASSES},
            "selected": selected,
        })

    return {
        "s3_uri": getattr(ref, "s3_uri", None),
        "bucket": getattr(ref, "bucket", None),
        "key": getattr(ref, "key", None),
        "station": ref.station, "date": ref.date, "chunk": ref.chunk,
        "fps": round(float(fps), 3), "n_frames": int(n_frames),
        "n_subvideos": len(subvideos),
        "subvideos": subvideos,
    }


def process_video(local_path, ref, backend, recipe: SplitRecipe,
                  k: int = 5, thr: float = THRESHOLD, every: int = 1) -> dict:
    fps, n_frames, frames = read_all_frames(local_path, every)
    if not frames:
        return {
            "s3_uri": getattr(ref, "s3_uri", None), "bucket": getattr(ref, "bucket", None),
            "key": getattr(ref, "key", None), "station": ref.station, "date": ref.date,
            "chunk": ref.chunk, "fps": round(float(fps), 3), "n_frames": int(n_frames),
            "n_subvideos": 0, "subvideos": [], "status": "no_frames",
        }
    return process_decoded(ref, fps, n_frames, frames, backend, recipe, k, thr)


def process_frames_chunk(group, backend, k: int = 5, thr: float = THRESHOLD) -> dict:
    """Process one ALREADY-SPLIT chunk (from ``frames_good``): per existing
    ``sub_NN`` do quality -> k diverse good -> day/night. No decode, no re-split."""
    import cv2

    subvideos = []
    for sub_id, items in group.subs.items():
        loaded = [(idx, p, cv2.imread(str(p), cv2.IMREAD_COLOR)) for idx, p in items]
        loaded = [(idx, p, im) for idx, p, im in loaded if im is not None]
        if not loaded:
            subvideos.append({"sub_id": sub_id, "start_frame": None, "end_frame": None,
                              "n_frames": 0, "boundary": None,
                              "quality": {c: 0 for c in _QUALITY_CLASSES}, "selected": []})
            continue
        idxs = [i for i, _, _ in loaded]
        paths = [p for _, p, _ in loaded]
        bgr = [im for _, _, im in loaded]

        labels = backend.score(bgr)
        counts = {c: 0 for c in _QUALITY_CLASSES}
        good = []                               # (position, confidence)
        for pos, (lab, conf) in enumerate(labels):
            counts[lab] = counts.get(lab, 0) + 1
            if lab == "good":
                good.append((pos, conf))

        selected = []
        if good:
            embeds = backend.embed([bgr[pos] for pos, _ in good])
            for rank, gi in enumerate(farthest_point_select(embeds, k), start=1):
                pos, conf = good[gi]
                dn, cval = day_night(bgr[pos], thr)
                selected.append({
                    "rank": rank, "frame_idx": idxs[pos], "timestamp_s": None,
                    "label": "good", "confidence": round(float(conf), 4),
                    "day_night": dn, "chroma": round(cval, 2) if cval is not None else None,
                    "file": paths[pos].name, "path": str(paths[pos]),
                })
            selected.sort(key=lambda r: r["rank"])

        subvideos.append({
            "sub_id": sub_id, "start_frame": min(idxs), "end_frame": max(idxs),
            "n_frames": len(idxs), "boundary": None,
            "quality": {c: counts.get(c, 0) for c in _QUALITY_CLASSES},
            "selected": selected,
        })

    return {
        "source": "frames", "station": group.station, "date": group.date,
        "chunk": group.chunk, "chunk_dir": str(group.chunk_dir),
        "n_subvideos": len(subvideos), "subvideos": subvideos,
    }


def build_frames_manifest(root, backend, *, k: int = 5, thr: float = THRESHOLD,
                          limit: int | None = None, on_video=None) -> dict:
    """Manifest from an already-split frame tree (``frames_good``)."""
    videos = []
    for i, group in enumerate(iter_chunk_groups(root)):
        if limit is not None and i >= limit:
            break
        try:
            rec = process_frames_chunk(group, backend, k, thr)
        except Exception as e:
            rec = {"source": "frames", "station": group.station, "date": group.date,
                   "chunk": group.chunk, "chunk_dir": str(group.chunk_dir),
                   "status": "error", "error": repr(e), "n_subvideos": 0, "subvideos": []}
        videos.append(rec)
        if on_video is not None:
            on_video(i, rec)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "frames", "frames_root": str(root),
        "k_per_subvideo": k, "daynight_thr": thr,
        "split_recipe": {"note": "pre-split input (frames already grouped into sub_NN)"},
        "n_videos": len(videos), "videos": videos,
    }


def build_manifest(source, backend, recipe: SplitRecipe | None = None, *,
                   date_from: str | None = None, date_to: str | None = None,
                   k: int = 5, thr: float = THRESHOLD, every: int = 1,
                   limit: int | None = None, on_video=None) -> dict:
    """Iterate ``source`` (yields VideoRefs), build the full manifest dict."""
    recipe = recipe or SplitRecipe()
    videos = []
    for i, ref in enumerate(source):
        if limit is not None and i >= limit:
            break
        try:
            with ref.open() as local_path:
                rec = process_video(local_path, ref, backend, recipe, k, thr, every)
        except Exception as e:                 # one bad video must not sink the run
            rec = {
                "s3_uri": getattr(ref, "s3_uri", None), "bucket": getattr(ref, "bucket", None),
                "key": getattr(ref, "key", None), "station": ref.station, "date": ref.date,
                "chunk": ref.chunk, "status": "error", "error": repr(e),
                "n_subvideos": 0, "subvideos": [],
            }
        videos.append(rec)
        if on_video is not None:
            on_video(i, rec)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "date_from": date_from, "date_to": date_to,
        "k_per_subvideo": k, "daynight_thr": thr, "sample_every": every,
        "split_recipe": recipe.as_dict(),
        "n_videos": len(videos),
        "videos": videos,
    }
