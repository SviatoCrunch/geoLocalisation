"""Full-taxonomy SegFormer-B3 training pipeline (isolated from segformer_seg).

Trains on all classes of a verified master taxonomy (SkyScenes 28-class candidate) from
source datasets (SkyScenes / FlyAwareV2 / Mid-Air), leakage-free split by town/scene, and
keeps ``gt_cramatorsc/GT_flat_mask`` as an untouched external validation set scored only on
railway / road / water. See docs/full_taxonomy_training.md. Nothing here mutates the existing
segformer_seg module.
"""
