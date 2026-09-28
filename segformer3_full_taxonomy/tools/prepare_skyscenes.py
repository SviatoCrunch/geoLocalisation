"""Flatten SkyScenes per-town tarballs into clean ``<out>/<kind>/<Town>/<id>.png`` dirs.

Each downloaded ``<Town>.tar.gz`` is a *plain* tar (mislabelled .gz) holding the town's ~69
real PNGs (2160x1440 RGBA) **directly**, plus ONE extra file that has a ``.png`` name but is
actually a redundant nested tar of the whole town. We extract the tar and keep only the files
whose bytes are real PNGs (magic ``\\x89PNG``), dropping the tar-masquerading-as-png.

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

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(8) == _PNG_MAGIC


def prepare_town(town_targz: Path, out_dir: Path) -> int:
    """Extract one town tar and keep only the REAL PNGs (flat). Returns count kept."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tempfile.TemporaryDirectory() as _td:
        td = Path(_td)
        subprocess.run(["tar", "xf", str(town_targz), "-C", str(td)], check=True)
        for p in td.rglob("*.png"):
            if _is_png(p):                       # real image; the fake tar-.png fails magic
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
