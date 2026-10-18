# video_label_pipeline

Self-contained geoLocalisation module for **video annotation**: from a date
range of videos, produce **one detailed JSON** describing how each video splits
into sub-videos and which frames to annotate.

```
videos (S3 or local, date_from…date_to)
  └─ split into sub-videos        content-adaptive SBD  (the frames_good recipe)
       └─ score every frame        quality_cls DINOv2 → good / usable / bad
            └─ pick K good frames   MAX VISUAL DIVERSITY (farthest-point on DINO) — not random
                 └─ tag each        day / night (chroma rule, thr 12)
  → manifest.json                   everything in JSON; no frames written to disk
```

## Why these choices
- **Split = content-adaptive**, the exact backend/recipe that built the
  `frames_good` tree (per its saved `structure.json`):
  `adaptive_thr 3.0 · min_content 15.0 · content_window 2 · content_reduce 4 · min_seg 2`.
  Ported verbatim from `quality-cls/tools/subvideo_homography.py` (`--backend
  content`), the author's recommended tool (the ORB-homography backends are a
  fallback that over-segments low-texture FPV).
- **Selection = farthest-point** on DINO embeddings among the `good` frames —
  deliberate, reproducible, spreads the chosen frames across the visual space
  instead of picking near-duplicates at random.
- **Day/night** reuses the existing chroma rule from
  `src_tif_build_universal/classify_night.py` (acc 0.986), per frame, label
  written to JSON (files are not renamed).

## Run
```bash
# S3 (mediamtx layout: live/<station>/filtered/<DD.MM.YYYY>/*.mp4)
python -m video_label_pipeline --source s3 --station 104,138 \
    --date-from 10.10.2026 --date-to 15.10.2026 --out manifest.json

# local folder of chunk_*.mp4
python -m video_label_pipeline --source local --dir /path/to/videos \
    --date-from 2026-10-10 --date-to 2026-10-15 --out manifest.json

# inspect the split WITHOUT the model (no torch/checkpoint needed)
python -m video_label_pipeline --source local --dir ./v \
    --date-from 2026-10-10 --date-to 2026-10-15 --dry-structure --out split.json
```

### Resume (long runs)
Every completed chunk is appended to `<out>.partial.jsonl` and flushed, so an
interrupted run **resumes** on restart — already-done chunks are skipped and new
ones appended (no progress lost; works for all three sources). Re-run the exact
same command to continue; pass `--fresh` to discard the partial and start over.
The final `<out>` JSON is assembled from the partial at the end.

### GPU on the server (Tesla T4, CUDA 12.4 driver)
The default `quality-cls` torch may be built for a newer CUDA than the driver
(→ "driver too old", CPU fallback, hours). Run with a matching wheel from the
geoLocalisation uv project instead (verified cu121 recipe):
```bash
cd /home/ubuntu/work/geoLocalisation
PYTHONPATH=/home/ubuntu/work/RevisitAnything/quality-cls/src \
uv run --extra-index-url https://download.pytorch.org/whl/cu121 \
    --with "torch==2.5.1+cu121" --with "torchvision==0.20.1" \
    --with opencv-python-headless --with numpy --with pandas --with pyyaml --with pillow \
    python -m video_label_pipeline --source frames \
    --dir /home/ubuntu/work/out/frames_good --k 5 \
    --out /home/ubuntu/work/out/frames_good/frames_good_manifest.json
```

## Dependency: quality_cls
The default backend loads `quality_cls.QualityPredictor` (frozen DINOv2 + linear
head) for **both** the quality label and the DINO embedding (one model load).
`quality_cls` lives in the RevisitAnything repo, so make it importable:
```bash
export PYTHONPATH=/path/to/RevisitAnything/quality-cls/src
```
and have its `checkpoints/head.pt` available (or pass `--checkpoint`). Only the
default backend needs torch; the split / diversity / day-night / manifest code
is pure and unit-tested with a fake backend.

## Output JSON (shape)
```json
{
  "generated_at": "...", "date_from": "10.10.2026", "date_to": "15.10.2026",
  "k_per_subvideo": 5, "daynight_thr": 12.0, "sample_every": 1,
  "split_recipe": {"tool": "subvideo_homography", "backend": "content",
                   "adaptive_thr": 3.0, "min_content": 15.0, "content_window": 2,
                   "content_reduce": 4, "min_seg": 2, "width": 480},
  "n_videos": 1,
  "videos": [{
    "s3_uri": "s3://mediamtx-recordings-crunch/live/104/filtered/10.10.2026/chunk_104_….mp4",
    "bucket": "mediamtx-recordings-crunch",
    "key": "live/104/filtered/10.10.2026/chunk_104_….mp4",
    "station": "104", "date": "10.10.2026", "chunk": "chunk_104_…",
    "fps": 30.0, "n_frames": 600, "n_subvideos": 3,
    "subvideos": [{
      "sub_id": "sub_01", "start_frame": 0, "end_frame": 123, "n_frames": 124,
      "boundary": null,
      "quality": {"good": 40, "usable": 30, "bad": 54},
      "selected": [
        {"rank": 1, "frame_idx": 22, "timestamp_s": 0.73,
         "label": "good", "confidence": 0.88, "day_night": "day", "chroma": 31.2}
      ]
    }]
  }]
}
```
`boundary` is `null` for the first sub-video; otherwise it records the cut frame
and the content delta that triggered it.

## Layout
| file | purpose |
|------|---------|
| `segment.py`   | content-adaptive shot-boundary split (ported, works on decoded frames) |
| `quality.py`   | `Backend` protocol + default `QualityClsBackend` (score + embed) |
| `diversity.py` | farthest-point selection of diverse frames |
| `daynight.py`  | chroma day/night (ported) |
| `video_io.py`  | OpenCV decode helpers |
| `source.py`    | `S3Source` / `LocalSource` + `VideoRef` (carries all prefixes) |
| `pipeline.py`  | orchestrator → manifest; `process_decoded` is the pure, testable core |
| `__main__.py`  | CLI |

## Tests
```bash
PYTHONPATH=. python -m pytest video_label_pipeline/tests -q      # no torch / S3 / GPU
```
