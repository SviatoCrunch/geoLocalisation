"""Convert a frame-level ``multicity_split.json`` into the ``{train,val,test:[point_id...]}``
split.json that :func:`build_split_relevance` (and thus ``export_shortlist``/``evaluate_ckpt``)
consumes.

``make_multicity_split.py`` writes ``{"frames": {"<city>:<stem>": {"split": ..., ...}}, "meta": ...}``.
The coarse tools instead read a flat ``{"train": [...], "val": [...], "test": [...]}`` list of
point_ids, where a point_id is exactly ``"<city>:<stem>"`` (see ``build_geo_points``). So this is a
pure regrouping: bucket each frame key by its ``split`` field. ``--city`` keeps only frames of the
given city (the rest are dropped from every split), which is what you want when the coarse config's
``gt`` is a single city — build_split_relevance would ignore the others anyway, but pruning keeps the
file honest and small.

Run::

    python -m geo_e2c_train.split_from_multicity \
        --in ~/work/out/gallery_h5/multicity_split.json \
        --city kup \
        --out ~/work/out/gallery_h5/split_kup.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_SPLITS = ("train", "val", "test")


def convert(frames: dict, city: str | None = None) -> dict:
    """Regroup ``{frame_key: {"split", "city", ...}}`` into ``{split: [frame_key...]}`` (sorted)."""
    out = {s: [] for s in _SPLITS}
    for key, fr in frames.items():
        if city is not None and fr.get("city") != city:
            continue
        sp = fr.get("split")
        if sp not in out:                       # ignore anything not in train/val/test
            continue
        out[sp].append(key)
    return {s: sorted(out[s]) for s in _SPLITS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="multicity_split.json (frame-level)")
    ap.add_argument("--city", default=None, help="keep only this city's frames (e.g. kup)")
    ap.add_argument("--out", required=True, help="destination split.json ({train,val,test:[ids]})")
    args = ap.parse_args(argv)

    data = json.loads(Path(args.inp).expanduser().read_text(encoding="utf-8"))
    frames = data.get("frames", data)          # tolerate a bare {frame_key: {...}} too
    split = convert(frames, args.city)

    Path(args.out).expanduser().write_text(json.dumps(split, indent=2), encoding="utf-8")
    n = {s: len(split[s]) for s in _SPLITS}
    scope = f"city={args.city}" if args.city else "all cities"
    print(f"[ok] {scope}: train {n['train']} / val {n['val']} / test {n['test']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
