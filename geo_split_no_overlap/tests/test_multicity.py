"""Per-city split: each city split independently (train/val/test) then unioned."""
import json
from pathlib import Path

import yaml

from geo_split_no_overlap import cli
from ._synth import write_h5, write_gt, xy_to_latlon, latlon_to_xy


def _city(tmp: Path, name: str, base_lat: float, base_lon: float, n: int):
    x0, y0 = latlon_to_xy(base_lat, base_lon)
    tiles, frames = [], []
    for i in range(n):
        x, y = x0 + i * 5000.0, y0                       # isolated -> one component each
        lat, lon = xy_to_latlon(x, y)
        tiles.append((f"{name}:t{i}", name, lat, lon, 1000.0))
        frames.append((f"f{i}", lat, lon))
    h5, gt = tmp / f"{name}.h5", tmp / f"gt_{name}"
    write_h5(h5, tiles)
    write_gt(gt, frames)
    return {"city": name, "gt": str(gt), "tiles_h5": str(h5)}


def _cfg(tmp: Path, cities, out: Path, tol=0.15) -> Path:
    c = {"cities": cities, "out_dir": str(out),
         "positive_selection": {"strategy": "current_rule", "params": {}},
         "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15,
         "ratio_tolerance": tol, "area_epsilon_m2": 1.0, "seed": 0}
    p = tmp / f"cfg_{out.name}.yaml"
    p.write_text(yaml.safe_dump(c), encoding="utf-8")
    return p


def _two_cities(tmp: Path, n=20):
    a = _city(tmp, "A", 48.5, 37.8, n)
    b = _city(tmp, "B", 49.5, 40.0, n)                   # far away -> galleries disjoint
    return [a, b]


def test_per_city_build_is_valid_and_balanced(tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(tmp_path, _two_cities(tmp_path), out)
    rc = cli.main(["build", "--config", str(cfg)])
    assert rc == 0

    data = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert data["status"] == "valid" and data["mode"] == "per_city"
    assert set(data["per_city"]) == {"A", "B"}
    for city, v in data["per_city"].items():
        assert v["within_tolerance"] is True
        # 20 points -> 14/3/3 = 0.7/0.15/0.15 exactly per city
        assert v["counts"]["train"] == 14 and v["counts"]["val"] == 3 and v["counts"]["test"] == 3

    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["cross_split_conflicts"] == 0

    # per-city subdirs written
    assert (out / "cities" / "A" / "split.json").exists()
    assert (out / "cities" / "B" / "split.json").exists()


def test_combined_is_union_of_per_city(tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(tmp_path, _two_cities(tmp_path), out)
    cli.main(["build", "--config", str(cfg)])
    combined = json.loads((out / "split.json").read_text(encoding="utf-8"))
    union_train = set()
    for city in ("A", "B"):
        sub = json.loads((out / "cities" / city / "split.json").read_text(encoding="utf-8"))
        union_train |= set(sub["train"])
    assert set(combined["train"]) == union_train


def test_per_city_determinism(tmp_path):
    c1, c2 = tmp_path / "r1", tmp_path / "r2"
    cities = _two_cities(tmp_path)
    cli.main(["build", "--config", str(_cfg(tmp_path, cities, c1))])
    cli.main(["build", "--config", str(_cfg(tmp_path, cities, c2))])
    s1 = json.loads((c1 / "split.json").read_text(encoding="utf-8"))
    s2 = json.loads((c2 / "split.json").read_text(encoding="utf-8"))
    for k in ("train", "val", "test"):
        assert s1[k] == s2[k]


def test_small_city_makes_run_infeasible(tmp_path):
    out = tmp_path / "run"
    cities = [_city(tmp_path, "A", 48.5, 37.8, 20),
              _city(tmp_path, "TINY", 50.5, 42.0, 2)]     # 2 points -> cannot fill 3 splits
    cfg = _cfg(tmp_path, cities, out)
    rc = cli.main(["build", "--config", str(cfg)])
    assert rc == 2
    data = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert data["status"] == "infeasible" and "TINY" in data["message"]


def test_per_city_kmz_export(tmp_path):
    out = tmp_path / "run"
    cfg = _cfg(tmp_path, _two_cities(tmp_path), out)
    cli.main(["build", "--config", str(cfg)])
    rc = cli.main(["export-kmz", "--config", str(cfg),
                   "--split", str(out / "split.json"), "--out", str(out / "viz.kmz"), "--tiles"])
    assert rc == 0
    import zipfile
    assert zipfile.is_zipfile(out / "viz.kmz")
