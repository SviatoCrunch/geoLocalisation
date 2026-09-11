"""vlad_vocab — fit K-means VLAD vocabularies + recast into the e2c dictionary blob.

Isolated (no RevisitAnything imports): reads the group-per-tile DINO galleries produced by
``map_extract`` (``<idx>_lvl0/ift_dino``), samples L2-normalised tokens, fits a self-contained
GPU K-means per K, and recasts each vocabulary into ONE ``k<K>.pt`` blob holding both

  * ``centroids``     (K, D)  — the classic/residual VLAD dictionary  ("vlad<K>"),
  * ``assign_weight`` (K, D)  — the SuperVLAD frozen assignment γ·L2(centroids) ("supervlad<K>"),

derived via the vendored ``siam_e2c_model.vendored.stage2_core.assignment_from_centroids`` (γ
calibrated on the sampled tokens). ``build_e2c_model(cfg, assign_path=k<K>.pt)`` loads
``assign_weight`` for the supervlad arm and ``centroids`` for the vlad/residual arm from this
single file.

Heavy deps (torch, h5py) are imported lazily so ``recast`` / tests import cheaply.

Run::

    python -m vlad_vocab.fit --h5 map_dinov2_kramatorsc_s250m_d1024.h5 \
        map_dinov2_kup_s250m_d1024.h5 map_dinov2_liman_day_s250m_d1024.h5 \
        --k 32 --out cache/e2c_dict --device cuda
    # -> cache/e2c_dict/k32.pt  (assign_weight + centroids, D=1024)
"""
