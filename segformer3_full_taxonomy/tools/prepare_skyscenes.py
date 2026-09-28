"""Un-nest SkyScenes' double-tar into flat per-town dirs (stdlib tarfile, streaming).

Each downloaded ``<Town>.tar.gz`` is a *plain* tar whose members are themselves tars, one per
frame; every nested tar holds the WHOLE town's real PNGs under a deep
``srv/.../<Town>/<id>.png`` path (plus a redundant nested ``<Town>.tar.gz`` we ignore). So we
open the outer tar, take ONE nested tar, and write its real PNGs flat to
``<out>/<kind>/<Town>/<id>.png`` — without materialising the ~2 GB of duplicate nested tars.

Run ON THE SERVER::

    python3 -m segformer3_full_taxonomy.tools.prepare_skyscenes \
        --download-root /home/ubuntu/work/datasets/SkyScenes \
        --condition H_35_P_0/ClearNoon \
        --out /home/ubuntu/work/datasets/SkyScenes/prepared/H_35_P_0_ClearNoon
"""
from __future__ import annotations

import argparse
import tarfile
import tempfile
from pathlib import Path


def _first_png_member(tf: tarfile.TarFile):
    """The first nested-tar member (a ``<id>.png`` that is really a tar) in an outer town tar."""
    for m in tf.getmembers():
        if m.isfile() and m.name.endswith(".png"):
            return m
    return None


def prepare_town(town_targz: Path, out_dir: Path) -> int:
    """Extract the real PNGs from one SkyScenes double-tar into ``out_dir`` (flat).

    Returns the number of PNGs written. Members ending ``.tar.gz`` (the redundant nested town
    tar) are skipped; every other ``*.png`` is a real image/mask and is written by basename.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(town_targz) as outer, tempfile.TemporaryDirectory() as td:
        inner_m = _first_png_member(outer)
        if inner_m is None:
            return 0
        # Extract the nested tar to a real file first — tarfile reads GNU @LongLink members
        # (deep long paths) reliably from a seekable file, but not from an extractfile() stream.
        outer.extract(inner_m, td, filter="data")
        inner_path = Path(td) / inner_m.name
        with tarfile.open(inner_path) as inner:
            for m in inner.getmembers():
                if not (m.isfile() and m.name.endswith(".png")):
                    continue  # skips the nested *.tar.gz
                with inner.extractfile(m) as f:
                    (out_dir / Path(m.name).name).write_bytes(f.read())
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
