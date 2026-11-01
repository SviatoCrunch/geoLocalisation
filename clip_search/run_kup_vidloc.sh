#!/usr/bin/env bash
# Whole-video geo-localization for Kupyansk labelled detections — motion-budget frame selection
# (place_in_motion, the production keyframe recipe) → coarse top-100 → fine MAGSAC++ with full
# per-level dump → per-video aggregation + detailed report.
#
# Runs the 6-stage chain end to end. Designed to be launched in the BACKGROUND on the GPU box:
#     cd ~/work/geoLocalisation
#     nohup bash clip_search/run_kup_vidloc.sh > ~/work/kup_vidloc/run.log 2>&1 &
#     tail -f ~/work/kup_vidloc/run.log
#
# Heavy stages need a GPU + a local sync of the packed kup_newformat store (~86 GiB) — this cannot
# run on a CPU-only / Windows dev box. Override any path via env vars below.
set -euo pipefail

# ---- paths (override via env) ----
CSV=${CSV:-$HOME/work/mapforml_public_Detection.csv}
OUT=${OUT:-$HOME/work/kup_vidloc}
ZONE=${ZONE:-0-17-34-138}
CITY=${CITY:-kup}
CKPT=${CKPT:-$HOME/work/out/runs_geo_iso_noverlap/run_best.pt}        # kup e2c SuperVLAD ckpt
ASSIGN=${ASSIGN:-$HOME/work/out/gallery_h5/dict/k32.pt}               # VLAD assign dict
GALLERY=${GALLERY:-$HOME/work/product_emb/kup_prodaction.h5}          # kup coarse cell gallery (221 tiles)
DATASET_DIR=${DATASET_DIR:-$HOME/work/mpp_local/kup}                  # local sync of kup_newformat
QUALITY_CKPT=${QUALITY_CKPT:-$HOME/work/RevisitAnything/quality-cls/checkpoints/head.pt}
QUALITY_SRC=${QUALITY_SRC:-$HOME/work/RevisitAnything/quality-cls/src}
TARGET_K=${TARGET_K:-8}
K=${K:-100}
DECODE_EVERY=${DECODE_EVERY:-1}
PYUV=${PYUV:-uv run --python 3.11 --extra-index-url https://download.pytorch.org/whl/cu121 --with torch==2.5.1+cu121 --with torchvision==0.20.1 --with opencv-python-headless --with scipy --with h5py --with numpy --with boto3 --with tqdm --with pillow}

mkdir -p "$OUT"
echo "[run] CSV=$CSV OUT=$OUT zone=$ZONE city=$CITY target_k=$TARGET_K k=$K"

# 1. motion-budget frame selection (production recipe: quality-gated place_in_motion)
if [ -f "$OUT/clips_manifest.json" ]; then
  echo "[run] stage 1 skipped (exists: $OUT/clips_manifest.json)"
else
  PYTHONPATH="$QUALITY_SRC" uv run --python 3.12 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --with torch==2.5.1+cu121 --with torchvision==0.20.1 --with opencv-python-headless \
    --with numpy --with pillow --with pyyaml --with pandas \
    python -m clip_search.sample_motion --csv "$CSV" --zone "$ZONE" --city "$CITY" \
      --out-dir "$OUT" --quality-gate good --target-k "$TARGET_K" --decode-every "$DECODE_EVERY" \
      --checkpoint "$QUALITY_CKPT"
fi

# 2. embed selected frames → query H5 (SAME backbone/proj as the kup gallery)
$PYUV python -m map_extract.extract --images "$OUT/frames" \
  --out "$OUT/query_d1024.h5" --backbone dinov2_vitg14 --desc_dim 1024 \
  --proj_seed 0 --segment_sky --device cuda --amp

# 3. coarse shortlist — top-K cells per frame
$PYUV python -m geo_e2c_train.coarse_rank_gallery --ckpt "$CKPT" --assign "$ASSIGN" \
  --gallery "$CITY=$GALLERY" --queries "$CITY=$OUT/query_d1024.h5" \
  --city "$CITY" --k "$K" --device cuda --out "$OUT/shortlist_k${K}.json"

# 4. fine MAGSAC++ rerank + FULL per-level dump (keep all K cells per frame)
$PYUV python -m pyramid_newformat.locate --queries "$CITY=$OUT/query_d1024.h5" \
  --indices "$OUT/shortlist_k${K}.json" --dataset-dir "$DATASET_DIR" \
  --topk "$K" --dump-levels --device cuda --out "$OUT/locate_k${K}.json"

# 5. per-video aggregation + distR + KMZ (keep top-100 aggregated cells)
$PYUV python -m clip_search.aggregate --locate "$OUT/locate_k${K}.json" \
  --manifest "$OUT/clips_manifest.json" --out-dir "$OUT/out" --topk "$K"

# 6. detailed report: per-frame top-100 with per-level scores + per-video distR
$PYUV python -m clip_search.report_levels --locate "$OUT/locate_k${K}.json" \
  --results "$OUT/out/clip_search_results.json" --manifest "$OUT/clips_manifest.json" \
  --out-dir "$OUT/out" --agg sum --topn 10 --topk-detail "$K"

echo "[done] report -> $OUT/out/report.csv + report.json ; KMZ -> $OUT/out/kmz/"
