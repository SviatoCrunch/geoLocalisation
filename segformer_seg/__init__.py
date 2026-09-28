"""Isolated SegFormer (MiT-B3) semantic-segmentation trainer for drone-frame scene parsing.

Self-contained module (lives in git and on the server under geoLocalisation/segformer_seg/).
Class taxonomy = 6-class union of the server GT masks (Railways/Road/Sky/Water) and the new
hand-labelled COCO set (adds bridge/tower); index 0 = background. See config.py.

Stages:
  1. build_manifest.py  — mix COCO + server masks (minus covered) into path-level JSONL.
  2. dataset.py         — compose index labels on the fly + augment (added stage 2).
  3. train.py / evaluate.py — HuggingFace SegformerForSemanticSegmentation (added stage 2).
"""
