"""End-to-end: synthetic gallery H5 + gt frames -> CLI build -> audit -> determinism."""
import json
from pathlib import Path

import yaml

from geo_split_no_overlap import cli
from ._synth import xy_to_latlon, latlon_to_xy, write_h5, write_gt


def _make_dataset(tmp: Path):
    x0, y0 = latlon_to_xy(48.5, 37.8)
    tiles, frames = [], []
    # 28 isolated tiles (5 km apart -> no overlap), one query each
    for i in range(28):
        x, y = x0 + i * 5000.0, y0
        lat, lon = xy_to_latlon(x, y)
        tiles.append((f"c:iso{i}", "c", lat, lon, 1000.0))
        frames.append((f"iso{i}", lat, lon))
    # one overlapping pair (stride 250 m) -> a single size-2 component
    for j, dx in enumerate((-6000.0, -5750.0)):
        x, y = x0 + dx, y0
        lat, lon = xy_to_latlon(x, y)
        tiles.append((f"c:ov{j}", "c", lat, lon, 1000.0))
        frames.append((f"ov{j}", lat, lon))

    h5 = tmp / "gallery.h5"
    gt = tmp / "gt_c"
    write_h5(h5, tiles)
    write_gt(gt, frames)
    return h5, gt


def _write_cfg(tmp: Path, h5: Path, gt: Path, out: Path, strategy="current_rule") -> Path:
    cfg = {
        "gt": [f"c={gt}"], "tiles_h5": str(h5), "out_dir": str(out),
        "positive_selection": {"strategy": strategy, "params": {}},
        "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15,
        "ratio_tolerance": 0.15, "area_epsilon_m2": 1.0, "seed": 0, "solver": "auto",
    }
    p = tmp / f"cfg_{strategy}_{out.name}.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def test_build_then_audit_valid(tmp_path):
    h5, gt = _make_dataset(tmp_path)
    out = tmp_path / "run"
    cfg = _write_cfg(tmp_path, h5, gt, out)

    rc = cli.main(["build", "--config", str(cfg)])
    assert rc == 0

    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["status"] == "valid"
    total = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == 30                                   # all points placed, none excluded
    assert (out / "components.json").exists()
    assert (out / "audit.json").exists()
    assert (out / "summary.md").exists()
    assert (out / "resolved_config.yaml").exists()

    # independent audit CLI must agree
    rc_audit = cli.main(["audit", "--config", str(cfg), "--split", str(out / "split.json")])
    assert rc_audit == 0

    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["cross_split_conflicts"] == 0


def test_determinism_same_seed(tmp_path):
    h5, gt = _make_dataset(tmp_path)
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    cli.main(["build", "--config", str(_write_cfg(tmp_path, h5, gt, out1))])
    cli.main(["build", "--config", str(_write_cfg(tmp_path, h5, gt, out2))])
    s1 = json.loads((out1 / "split.json").read_text(encoding="utf-8"))
    s2 = json.loads((out2 / "split.json").read_text(encoding="utf-8"))
    for k in ("train", "val", "test"):
        assert s1[k] == s2[k]


def test_dry_run_writes_nothing(tmp_path):
    h5, gt = _make_dataset(tmp_path)
    out = tmp_path / "dry"
    cfg = _write_cfg(tmp_path, h5, gt, out)
    rc = cli.main(["build", "--config", str(cfg), "--dry-run"])
    assert rc == 0
    assert not (out / "split.json").exists()


def test_swap_strategy_via_yaml_only(tmp_path):
    # changing ONLY the strategy in YAML produces a valid split with no code change
    h5, gt = _make_dataset(tmp_path)
    out = tmp_path / "cp"
    cfg = _write_cfg(tmp_path, h5, gt, out, strategy="contains_point")
    rc = cli.main(["build", "--config", str(cfg)])
    assert rc == 0
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    assert split["status"] == "valid"
    assert split["positive_selection"]["strategy"] == "contains_point"


def test_list_positive_strategies(capsys):
    rc = cli.main(["list-positive-strategies"])
    assert rc == 0
    names = capsys.readouterr().out.split()
    assert "current_rule" in names and "contains_point" in names
