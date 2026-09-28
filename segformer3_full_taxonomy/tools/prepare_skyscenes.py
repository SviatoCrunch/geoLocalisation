"""Un-nest SkyScenes' double-tar into flat per-town dirs (via system ``tar``).

Each downloaded ``<Town>.tar.gz`` is a *plain* tar whose members are themselves tars, one per
frame; every nested tar holds the WHOLE town's real PNGs under a deep
``srv/.../<Town>/<id>.png`` path (plus a redundant nested ``<Town>.tar.gz`` we ignore). Python's
``tarfile`` cannot read these nested GNU-``@LongLink`` archives, but the system ``tar`` reads
them fine — so we shell out to ``tar`` for both levels and flatten the real PNGs to
``<out>/<kind>/<Town>/<id>.png``.

Run ON THE SERVER (needs ``tar`` on PATH)::

    python3 -m segformer3_full_taxonomy.tools.prepare_skyscenes \
        --download-root /home/ubuntu/work/datasets/SkyScenes \
        --condition H_35_P_0/ClearNoon \
        --out /home/ubuntu/work/datasets/SkyScenes/prepared/H_35_P_0_ClearNoon
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def prepare_town(town_targz: Path, out_dir: Path) -> int:
    """Extract the real PNGs from one SkyScenes double-tar into ``out_dir`` (flat).

    Returns the number of PNGs written. Uses ``tar`` twice (outer -> one nested tar -> real
    PNGs) and flattens; the redundant nested ``<Town>.tar.gz`` is never a ``*.png`` so it is
    naturally skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run(["tar", "tf", str(town_targz)], capture_output=True, text=True).stdout
    inner_name = next((ln.strip() for ln in listing.splitlines() if ln.strip().endswith(".png")), None)
    if not inner_name:
        return 0
    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        subprocess.run(["tar", "xf", str(town_targz), "-C", str(td), inner_name], check=True)
        inner_path = next(td.rglob("*.png"), None)   # the nested tar (named <id>.png)
        if inner_path is None:
            return 0
        ext = td / "ext"
        ext.mkdir()
        subprocess.run(["tar", "xf", str(inner_path), "-C", str(ext)], check=True)
        n = 0
        for p in ext.rglob("*.png"):                 # real PNGs at a deep srv/.../<Town>/ path
            shutil.move(str(p), str(out_dir / p.name))
            n += 1
    return n


def run(download_root: Path, condition: str, out: Path, log=print) -> dict:
    report: dict = {"condition": condition, "kinds": {}}
    for kind in ("Images", "Segment"):
        base = download_root / kind / condition
        if not base.is_dir():
            log(f"[skip] {base} (not found)")
            continue
        per_town = {}
        for town_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            targz = town_dir / f"{town_dir.name}.tar.gz"
            if not targz.exists():
                continue
            n = prepare_town(targz, out / kind / town_dir.name)
            per_town[town_dir.name] = n
            log(f"[{kind}] {town_dir.name}: {n} png")
        report["kinds"][kind] = per_town
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--download-root", required=True, help="SkyScenes root (has Images/ Segment/)")
    ap.add_argument("--condition", required=True, help="e.g. H_35_P_0/ClearNoon")
    ap.add_argument("--out", required=True, help="flat output root (<out>/Images/<Town>/*.png ...)")
    args = ap.parse_args(argv)

    rep = run(Path(args.download_root).expanduser(), args.condition, Path(args.out).expanduser())
    for kind, towns in rep["kinds"].items():
        print(f"== {kind}: {sum(towns.values())} png over {len(towns)} towns ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
