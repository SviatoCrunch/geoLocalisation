"""Fetch + prepare + cleanup a list of SkyScenes conditions (download -> prepare -> rm raw tar).

Each condition is ``HP/Weather`` (e.g. ``H_35_P_45/ClearNoon``, ``H_35_P_0/ClearSunset``). Images
are downloaded always; Segment only for ClearNoon (masks are weather-invariant, so weather
Images pair to the same-HP ClearNoon masks at train time). Only ``prepared/<HP>_<Weather>/`` is
kept — the ~13 GB raw tarballs are deleted after prepare, so many conditions fit on disk.

IMPORTANT: include each HP's ``ClearNoon`` in the list so its weather variants have masks.

    python3 -m segformer3_full_taxonomy.tools.fetch_skyscenes \
      --download-root /home/ubuntu/work/datasets/SkyScenes \
      --prepared-dir  /home/ubuntu/work/datasets/SkyScenes/prepared \
      --conditions H_35_P_0/ClearNoon H_35_P_45/ClearNoon H_35_P_60/ClearNoon H_35_P_90/ClearNoon \
                   H_35_P_0/ClearSunset H_35_P_0/CloudyNoon H_35_P_0/MidRainyNoon
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_REPO = "hoffman-lab/SkyScenes"


def fetch_one(download_root: Path, prepared_dir: Path, cond: str, log=print) -> int:
    from huggingface_hub import snapshot_download

    from .prepare_skyscenes import run as prepare_run

    hp, weather = cond.split("/")
    patterns = [f"Images/{hp}/{weather}/*"]
    if weather == "ClearNoon":
        patterns.append(f"Segment/{hp}/{weather}/*")
    log(f"[fetch] {cond} ...")
    snapshot_download(_REPO, repo_type="dataset", local_dir=str(download_root), allow_patterns=patterns)
    out = prepared_dir / f"{hp}_{weather}"
    rep = prepare_run(download_root, cond, out, log=log)
    n = sum(sum(t.values()) for t in rep["kinds"].values())
    for kind in ("Images", "Segment"):                    # drop raw tarballs, keep prepared/
        shutil.rmtree(download_root / kind / hp / weather, ignore_errors=True)
    log(f"[done] {cond}: {n} png -> {out}")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download-root", required=True)
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--conditions", nargs="+", required=True, help="HP/Weather e.g. H_35_P_45/ClearNoon")
    args = ap.parse_args(argv)

    dr = Path(args.download_root).expanduser()
    pd = Path(args.prepared_dir).expanduser()
    total = 0
    for cond in args.conditions:
        total += fetch_one(dr, pd, cond)
    print(f"== fetched+prepared {len(args.conditions)} conditions, {total} png -> {pd} ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
