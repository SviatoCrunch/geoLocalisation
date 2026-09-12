"""Synthetic fixtures for geo_train_batching tests."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

_R = 6378137.0


def xy_to_latlon(x, y):
    return (math.degrees(2.0 * math.atan(math.exp(y / _R)) - math.pi / 2.0),
            math.degrees(x / _R))


def latlon_to_xy(lat, lon):
    return (_R * math.radians(lon),
            _R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def rand_problem(B=6, M=24, seed=0, max_pos=3):
    """Random (scores, pos_mask, cand_mask) with pos ⊆ cand and >=1 positive per row."""
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, M, generator=g)
    pos = torch.zeros(B, M, dtype=torch.bool)
    cand = torch.zeros(B, M, dtype=torch.bool)
    rng = np.random.RandomState(seed)
    for i in range(B):
        npos = rng.randint(1, max_pos + 1)
        p = rng.choice(M, size=npos, replace=False)
        pos[i, p] = True
        # safe negatives: some other columns
        rest = [c for c in range(M) if not pos[i, c]]
        nsafe = rng.randint(1, len(rest) + 1)
        s = rng.choice(rest, size=nsafe, replace=False)
        cand[i] = pos[i].clone()
        cand[i, s] = True
    return S, pos, cand


def rand_bb_problem(B=6, seed=0):
    """Random B×B (S, R_pos, R_cand) with a positive diagonal and R_cand ⊇ R_pos."""
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(B, B, generator=g)
    rng = np.random.RandomState(seed)
    R_pos = torch.eye(B, dtype=torch.bool)
    R_cand = R_pos.clone()
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            r = rng.rand()
            if r < 0.2:
                R_pos[i, j] = True
                R_cand[i, j] = True
            elif r < 0.6:
                R_cand[i, j] = True
    return S, R_pos, R_cand


# ── inputs for the split adapter integration test ──────────────────────────────────
def write_h5(path: Path, tiles):
    import h5py
    sdt = h5py.string_dtype("utf-8")
    n = len(tiles)
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=np.array([t[2] for t in tiles], float))
        f.create_dataset("lon", data=np.array([t[3] for t in tiles], float))
        f.create_dataset("window_size_m", data=np.array([t[4] for t in tiles], float))
        f.create_dataset("tile_id", data=np.array([t[0] for t in tiles], dtype=object), dtype=sdt)
        f.create_dataset("city", data=np.array([t[1] for t in tiles], dtype=object), dtype=sdt)
        f.create_dataset("vlad", data=np.zeros((n, 4), np.float16))


def write_gt(root: Path, frames):
    root.mkdir(parents=True, exist_ok=True)
    for fid, lat, lon in frames:
        (root / f"{fid}_{lat}_{lon}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def make_split_dataset(tmp: Path, n=30, strategy="current_rule", params=None):
    """Isolated single-city gallery + queries; returns (config_path, split_json_path)."""
    import yaml
    from geo_split_no_overlap import cli
    x0, y0 = latlon_to_xy(48.5, 37.8)
    tiles, frames = [], []
    for i in range(n):
        x, y = x0 + i * 5000.0, y0
        lat, lon = xy_to_latlon(x, y)
        tiles.append((f"c:t{i}", "c", lat, lon, 1000.0))
        frames.append((f"f{i}", lat, lon))
    h5, gt, out = tmp / "g.h5", tmp / "gt_c", tmp / "run"
    write_h5(h5, tiles)
    write_gt(gt, frames)
    cfg = {"gt": [f"c={gt}"], "tiles_h5": str(h5), "out_dir": str(out),
           "positive_selection": {"strategy": strategy, "params": params or {}},
           "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15,
           "ratio_tolerance": 0.15, "area_epsilon_m2": 1.0, "seed": 0}
    cfgp = tmp / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = cli.main(["build", "--config", str(cfgp)])
    assert rc == 0
    return cfgp, out / "split.json"
