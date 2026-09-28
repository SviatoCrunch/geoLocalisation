"""Build the path-level training manifest: new COCO labels + server GT masks (minus covered).

Mixes the two mask sources WITHOUT merging folders — it only records paths (+ inline COCO
polygons) into JSONL, exactly as asked. Sources:

  * **server GT** ``gt_cramatorsc``: one entry per GT still that has masks, read from
    ``gt_mask_audit.csv`` (already the clean, lat_lon-resolved view: per-N ``png_classes``).
    Per-class PNG paths are derived from the still's ``<lat>_<lon>`` + raw class name.
    Stills whose N is in ``covered_gt.csv`` (already hand-labelled in the new COCO set) are
    EXCLUDED so the same frame is not supervised twice.
  * **new COCO** (Roboflow export ``segment.v1i.coco``): one entry per labelled frame, its
    polygons grouped by canonical class and stored inline.

Everything the build had to drop is logged to ``build_report.json`` (covered-excluded,
missing PNGs, stills-without-mask, unresolved uuid-N orphans, dropped Building/Target), so a
capped/skipped item never reads as "covered". A train-split pixel histogram is written for
class-balanced loss weights.

Run ON THE SERVER (files live there)::

    python3 -m segformer_seg.build_manifest \
      --gt-root  /home/ubuntu/work/gt_cramatorsc \
      --coco-dir /home/ubuntu/work/gt_cramatorsc/new_coco/train \
      --covered  /home/ubuntu/work/gt_cramatorsc/covered_gt.csv \
      --out-dir  /home/ubuntu/work/geoLocalisation/segformer_seg/manifests \
      --val-frac 0.1 --test-frac 0.1 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .config import CLASSES, NUM_CLASSES, canonical
from .label_maps import compose_from_coco, compose_from_pngs

_STILL_RE = re.compile(r"^(\d+)_([0-9.]+)_([0-9.]+)\.jpg$")


# ----------------------------- source: server GT masks -----------------------------

def _covered_ns(covered_csv: Path | None) -> set[int]:
    if not covered_csv or not covered_csv.exists():
        return set()
    ns = set()
    with covered_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ns.add(int(row["n"]))
    return ns


def server_entries(gt_root: Path, covered: set[int], report: dict) -> list[dict]:
    """One entry per GT still with masks (minus covered). Paths derived, existence checked."""
    audit = gt_root / "gt_mask_audit.csv"
    mask_dir = gt_root / "GT_flat_mask"
    flat_dir = gt_root / "GT_flat"
    entries, excluded, no_mask, missing = [], [], [], []
    with audit.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = int(row["N"])
            still = row["gt_flat"]
            m = _STILL_RE.match(still)
            if not m:
                continue
            if row.get("has_mask", "0") not in ("1", "True", "true") or not row.get("png_classes"):
                no_mask.append(n)
                continue
            if n in covered:
                excluded.append(n)
                continue
            lat, lon = m.group(2), m.group(3)
            items = []
            for raw in row["png_classes"].split(";"):
                raw = raw.strip()
                if not raw or canonical(raw) is None:
                    continue
                png = mask_dir / f"{lat}_{lon}__{raw}.png"
                if not png.exists():
                    missing.append(png.name)
                    continue
                items.append([canonical(raw), str(png)])
            img = flat_dir / still
            if not items or not img.exists():
                if not img.exists():
                    missing.append(still)
                continue
            entries.append({"id": f"server:{n}", "source": "server", "image": str(img),
                            "height": 480, "width": 640, "label_pngs": items})
    report["server"] = {"used": len(entries), "excluded_covered": sorted(excluded),
                        "stills_without_mask": sorted(no_mask), "missing_files": sorted(set(missing))}
    return entries


# ------------------------------- source: new COCO -------------------------------

def coco_entries(coco_dir: Path, report: dict) -> list[dict]:
    """One entry per labelled COCO frame, polygons grouped by canonical class (inline)."""
    ann_path = coco_dir / "_annotations.coco.json"
    d = json.loads(ann_path.read_text(encoding="utf-8"))
    cats_by_id = {c["id"]: c["name"] for c in d["categories"]}
    anns_by_img: dict[int, list] = {}
    for a in d["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    dropped_cats = Counter()
    entries, missing, empty = [], [], []
    for im in d["images"]:
        img = coco_dir / im["file_name"]
        if not img.exists():
            missing.append(im["file_name"])
            continue
        by_class: dict[str, list] = {}
        for a in anns_by_img.get(im["id"], []):
            raw = cats_by_id.get(a["category_id"], "")
            cls = canonical(raw)
            if cls is None:
                dropped_cats[raw] += 1
                continue
            for poly in (p for p in a.get("segmentation", []) if isinstance(p, list)):
                by_class.setdefault(cls, []).append(poly)
        if not by_class:
            empty.append(im["file_name"])
            continue
        items = [[cls, polys] for cls, polys in by_class.items()]
        entries.append({"id": f"coco:{im['id']}", "source": "coco", "image": str(img),
                        "height": int(im["height"]), "width": int(im["width"]),
                        "label_polys": items})
    report["coco"] = {"used": len(entries), "dropped_categories": dict(dropped_cats),
                      "missing_files": sorted(missing), "empty_after_drop": sorted(empty)}
    return entries


# ------------------------------- label + histogram -------------------------------

def compose_label(entry: dict) -> np.ndarray:
    size = (entry["height"], entry["width"])
    if entry["source"] == "server":
        return compose_from_pngs([(c, p) for c, p in entry["label_pngs"]], size)
    cats = {i: c for i, (c, _) in enumerate(entry["label_polys"])}
    anns = [{"category_id": i, "segmentation": polys} for i, (_, polys) in enumerate(entry["label_polys"])]
    return compose_from_coco(anns, cats, size)


def _progress(seq, desc: str):
    """tqdm progress if available; else a dependency-free \\r counter."""
    seq = list(seq)
    try:
        from tqdm import tqdm
        yield from tqdm(seq, desc=desc, unit="img")
        return
    except Exception:
        n = len(seq)
        for i, x in enumerate(seq, 1):
            if i % 20 == 0 or i == n:
                print(f"\r  {desc}: {i}/{n}", end=("" if i < n else "\n"), flush=True)
            yield x


def pixel_histogram(entries: list[dict]) -> np.ndarray:
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    for e in _progress(entries, "hist"):
        hist += np.bincount(compose_label(e).ravel(), minlength=NUM_CLASSES)
    return hist


# ----------------------------------- split -----------------------------------

def split(entries: list[dict], val_frac: float, test_frac: float, seed: int) -> dict[str, list[dict]]:
    idx = np.arange(len(entries))
    np.random.default_rng(seed).shuffle(idx)
    n_val = int(round(len(entries) * val_frac))
    n_test = int(round(len(entries) * test_frac))
    val = {int(i) for i in idx[:n_val]}
    test = {int(i) for i in idx[n_val:n_val + n_test]}
    out = {"train": [], "val": [], "test": []}
    for i, e in enumerate(entries):
        out["val" if i in val else "test" if i in test else "train"].append(e)
    return out


# ----------------------------------- main -----------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", required=True, help="gt_cramatorsc root")
    ap.add_argument("--coco-dir", required=True, help="COCO train dir (has _annotations.coco.json + jpgs)")
    ap.add_argument("--covered", default=None, help="covered_gt.csv (N already labelled in COCO -> excluded)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-hist", action="store_true", help="skip train pixel histogram (faster)")
    args = ap.parse_args(argv)

    gt_root = Path(args.gt_root).expanduser()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    covered = _covered_ns(Path(args.covered).expanduser() if args.covered else None)
    ent = server_entries(gt_root, covered, report) + coco_entries(Path(args.coco_dir).expanduser(), report)
    print(f"[entries] server+coco = {len(ent)} "
          f"(server {report['server']['used']}, coco {report['coco']['used']})", flush=True)

    parts = split(ent, args.val_frac, args.test_frac, args.seed)
    for name, rows in parts.items():
        p = out / f"manifest_{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[split] {name}: {len(rows)} -> {p.name}", flush=True)
    report["split"] = {k: len(v) for k, v in parts.items()}
    report["covered_excluded"] = sorted(covered)

    if not args.no_hist:
        hist = pixel_histogram(parts["train"])
        total = int(hist.sum())
        freq = (hist / max(total, 1)).tolist()
        # median-frequency balancing (Eigen&Fergus) — robust class weights
        nz = hist[hist > 0]
        med = float(np.median(nz / total)) if len(nz) else 0.0
        mfb = [float(med / f) if f > 0 else 0.0 for f in freq]
        report["train_pixel_hist"] = {"classes": CLASSES, "counts": hist.tolist(),
                                      "freq": freq, "median_freq_weights": mfb}
        print("[hist] train pixels per class:")
        for c, n, w in zip(CLASSES, hist.tolist(), mfb):
            print(f"    {c:11} {n:>12,d}  w={w:.3f}", flush=True)

    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] manifests + build_report.json -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
