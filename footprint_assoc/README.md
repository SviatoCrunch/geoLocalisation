# footprint_assoc

Zero-shot association of a UAV frame with the correct level/footprint of its **concentric
satellite pyramid** (the frame's map centre is already known). Output is used to build
**overlap-aware positives + weights** for the main training — this is **not** global
geolocation. Fully self-contained: core is numpy-only; the heavy DINOv2/COG pieces are
**injected** via protocols, so estimators/fusion/association/KMZ are testable without a GPU.

## Problem in one line
For each frame, pick the pyramid scale `s*` (or a soft distribution over scales, or
*refuse*) whose satellite crop best matches the actually-visible ground area, then convert
that footprint into graded links to gallery tiles (CQ/CT/IoU).

## Pipeline

```python
from footprint_assoc import FootprintConfig, pyramid, kmz
from footprint_assoc.pipeline import estimate_and_associate
from footprint_assoc.extractors.dinov2 import DinoV2Extractor   # server-only (torch)

cfg = FootprintConfig()                       # scales 100..1000 (r=1.21), cascade fusion
pyr = pyramid.build(frame_id, lat, lon, uav_img, cfg.scales(), crop_source, DinoV2Extractor())
est, pairs = estimate_and_associate(pyr, gallery_tiles, cfg)   # tiles: (id, lat, lon, size_m)
kmz.write_kmz("frame.kmz", [est], tiles_geom={...}, assoc={est.frame_id: pairs})
```

`estimate_frame` returns a `FrameEstimate`: per-method level scores, a **soft distribution**
over scales, `best_scale` (or `None`), `confidence`, and `status` ∈ {`accept`,`soft`,`refuse`}.

## Iteration 1 — estimators (data-ready, on frozen DINOv2 tokens)
- **`global_vlad`** — `Sglobal(s)=cos(D(Q),D(G(c,s)))` (AnyLoc-style). Cheap; keeps the whole
  curve; risk = cosine scale-bias.
- **`patch_overlap`** — `Spatch(s)=trimmed_mean(match_sims)·√(CQ·CG)` over mutual-NN patch
  correspondences; robust to context; risk = repetitive texture.

Both are `Strategy + Registry` (`create_estimator`); add more with `register_estimator`.
Geometry (RANSAC → polygon) and semantic anchors are later iterations.

## Fusion — cascade (default) + rank_vote (experiment)
- **`cascade`**: global → top candidates; patch → orders them; **agreement between the two
  drives confidence, not the pick**. Weak margin or method disagreement → `soft`; no support
  → `refuse`. Never forces argmax.
- **`rank_vote`**: scale-invariant Reciprocal-Rank Fusion — provided for A/B only.

**On per-level voting (the design question):** naive equal-weight voting is *not* the
primary selector here — the two DINOv2 estimators are **correlated** (shared scale-bias), so
voting doesn't cancel their common error, and different score scales aren't comparable. So
voting is used as an **agreement/confidence signal** (independent methods agreeing on the
same/adjacent level ⇒ high confidence; disagreement ⇒ soft/undetermined), plus an optional
**rank-fusion** experiment. When geometry (RANSAC) is added it will be a **veto/verifier**,
not an equal vote.

## Footprint → gallery association
`associate(estimate, tiles, cfg)` → per tile `CQ = |P∩T|/|P|`, `CT = |P∩T|/|T|`, `IoU`, using
the **expected** overlap over levels (Σ_s p(s)·metric) so soft estimates give graded weights.
Status: `strong` (high CQ+CT) · `partial` (IoU≥thr) · `negative` · `undetermined` (a *refused*
frame yields only undetermined/negative — never hard labels).

## KMZ (poor-man's oracle)
No ground-truth footprint exists, so `kmz.write_kmz([...])` draws, per frame: the concentric
pyramid squares (fill opacity ∝ fused probability, chosen scale highlighted), the centre
coloured by status, and gallery tiles coloured by pair status — open in Google Earth to
eyeball picks and where methods disagree.

## Isolation
numpy-only core; the DINOv2 extractor + satellite crop reader are behind
`extractors.base` protocols (real DINOv2 adapter in `extractors/dinov2.py`, torch, server-
only, imported lazily). No dependency on RevisitAnything or sibling packages.

## Tests
```bash
python -m pytest footprint_assoc -q
```
Geometry, estimators (peak at the true level), fusion (accept/soft/refuse, rrf), association
(strong/partial/negative/undetermined), pyramid build via a fake extractor, and KMZ.
