# vlad_vocab — VLAD/SuperVLAD K=32 dictionary for the e2c model (isolated)

Fits a K-means VLAD vocabulary on the DINO galleries (`map_extract` output) and recasts it
into ONE `k<K>.pt` blob holding **both** dictionaries the e2c model needs:

| key | shape | used by | "name" |
|---|---|---|---|
| `centroids` | (K, D) | vlad / residual arm (hard-argmin residual) | **vlad{K}** |
| `assign_weight` | (K, D) | supervlad arm (frozen softmax assignment) | **supervlad{K}** |

`assign_weight = γ · L2(centroids)`, with γ calibrated on the sampled tokens via the vendored
`siam_e2c_model.vendored.stage2_core.assignment_from_centroids` (single source of truth). So
one fit produces both "vlad32" and "supervlad32", and `build_e2c_model(cfg, assign_path=k32.pt)`
loads `assign_weight` for the supervlad arm and `centroids` for the vlad/residual arm from the
same file.

> Fit on the **current** tokens: after re-extracting at `--desc_dim 1024`, D = 1024, so old
> 840/1536-dim dictionaries are incompatible — refit here and set `E2cModelConfig.d_token=1024`.

## Run (after the 3 galleries are extracted)

```bash
cd ~/work/geoLocalisation
uv run --python 3.11 --with "torch==2.5.1" --with h5py --with numpy --with tqdm \
  python -m vlad_vocab.fit \
  --h5 /home/ubuntu/work/out/gallery_h5/map_dinov2_kramatorsc_s250m_d1024.h5 \
       /home/ubuntu/work/out/gallery_h5/map_dinov2_kup_s250m_d1024.h5 \
       /home/ubuntu/work/out/gallery_h5/map_dinov2_liman_day_s250m_d1024.h5 \
  --k 32 --out /home/ubuntu/work/out/e2c_dict --device cuda
```

Writes `k32.pt` (assign_weight + centroids), `c32_centers.pt` (raw centroids), `k32.meta.json`.
Add more K with `--k 32 64 128` (one gallery read pass fits all).

## Files / deps

`h5_tokens.py` (reader+sampler, byte-identical normalisation to the VLAD build), `kmeans.py`
(self-contained Lloyd, no faiss), `recast.py` (centroids→blob via vendored core), `fit.py`
(CLI). torch/h5py imported lazily. Tests: `python -m pytest vlad_vocab/tests` (K-means
convergence, recast blob round-trips through the e2c loaders, H5 reader; h5py-gated).
