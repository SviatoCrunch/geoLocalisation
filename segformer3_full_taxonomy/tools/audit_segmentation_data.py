"""Audit segmentation datasets BEFORE any training — the §1 data audit.

Config-driven (YAML/JSON), no torch (numpy + PIL only), so it runs anywhere and is the first
thing to run on a freshly-downloaded source or on the external-val set. It NEVER invents
palettes or IDs — it only reports what is actually on disk, so the master-taxonomy mapping
can be authored from verified facts.

Per source it reports:
  * image + mask counts, resolutions, formats;
  * unique mask values (single-channel IDs) OR unique RGB colours, with per-value pixel counts;
  * image<->mask pairing, with unmatched / corrupt / empty lists (nothing silently dropped);
  * SHA-256 of the sorted file lists (so splits/manifests are verifiable and reproducible).

Two mask layouts:
  * ``single``            — one label mask per image (``format: index|rgb|auto``);
  * ``per_class_binary``  — several binary PNGs per image, one per class (the gt_cramatorsc
                            ``GT_flat_mask`` external-val layout).

Outputs ``audit.json`` (machine-readable) + ``audit.md`` (human summary).

Run ON THE SERVER::

    python3 -m segformer3_full_taxonomy.tools.audit_segmentation_data \
        --config segformer3_full_taxonomy/configs/audit_example.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


# ------------------------------- file discovery -------------------------------

def _find(root: str, glob: str) -> list[Path]:
    return sorted(Path(root).expanduser().rglob(glob) if "**" in glob
                  else Path(root).expanduser().glob(glob))


def _sha256_of_names(paths: list[Path], root: str) -> str:
    r = Path(root).expanduser()
    h = hashlib.sha256()
    for p in sorted(str(p.relative_to(r)) for p in paths):
        h.update(p.encode("utf-8")); h.update(b"\n")
    return h.hexdigest()


# ------------------------------- mask reading -------------------------------

def _detect_format(mask_path: Path) -> str:
    from PIL import Image
    im = Image.open(mask_path)
    return "rgb" if im.mode in ("RGB", "RGBA") else "index"


def _mask_stats(mask_path: Path, fmt: str, hist: Counter, res: Counter) -> None:
    """Accumulate unique value/colour pixel counts + resolution for one mask."""
    from PIL import Image
    im = Image.open(mask_path)
    res[im.size] += 1  # (W, H)
    if fmt == "rgb":
        a = np.asarray(im.convert("RGB")).reshape(-1, 3).astype(np.int64)
        # pack RGB into one int for fast counting (int64 so <<8/<<16 don't overflow uint8)
        packed = (a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2]
        vals, cnts = np.unique(packed, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            hist[(v >> 16 & 255, v >> 8 & 255, v & 255)] += c
    else:
        a = np.asarray(im.convert("L") if im.mode not in ("L", "P", "I") else im)
        vals, cnts = np.unique(a, return_counts=True)
        for v, c in zip(vals.tolist(), cnts.tolist()):
            hist[int(v)] += c


# ------------------------------- source audits -------------------------------

def _pair_key(p: Path, pair_key: str | None) -> str:
    """Pairing key for an image/mask file. ``pair_key`` = regex whose group 1 is the shared id
    (e.g. ``(\\d+)`` pairs SkyScenes ``008261_clrnoon.png`` with ``008261_semsegCarla…png``).
    None -> the filename stem."""
    if pair_key:
        m = re.search(pair_key, p.name)
        return m.group(1) if m else p.stem
    return p.stem


def audit_single(images: list[Path], masks: list[Path], fmt: str, do_hist: bool,
                 pair_key: str | None = None, palette: dict | None = None) -> dict:
    """One label mask per image. The palette histogram is scanned over ALL masks (independent
    of pairing); pairing (image<->mask) is a separate set match on ``pair_key``. If ``palette``
    ({(r,g,b): name}) is given for an rgb mask, pixel counts are reported per class name."""
    if fmt == "auto" and masks:
        fmt = _detect_format(masks[0])
    hist: Counter = Counter()
    res: Counter = Counter()
    corrupt, empty = [], []
    if do_hist:
        for m in masks:
            try:
                before = sum(hist.values())
                _mask_stats(m, fmt, hist, res)
                if sum(hist.values()) == before:
                    empty.append(m.name)
            except Exception as e:  # noqa: BLE001 — record corrupt, keep going
                corrupt.append({"file": m.name, "error": str(e)[:200]})
    img_keys = {_pair_key(i, pair_key) for i in images}
    msk_keys = {_pair_key(m, pair_key) for m in masks}
    matched = len(img_keys & msk_keys)
    return _summ("single", fmt, images, masks, matched, hist, res,
                 sorted(img_keys - msk_keys), sorted(msk_keys - img_keys), corrupt, empty,
                 palette=palette)


def audit_per_class_binary(images: list[Path], mask_root: str, key: str, do_hist: bool) -> dict:
    """Several binary PNGs per image (``<key>__<Class>.png``); key derived from image name."""
    mask_root = Path(mask_root).expanduser()
    all_masks = sorted(mask_root.glob("*.png"))
    cls_counter: Counter = Counter()      # class name -> #images that have it
    px_counter: Counter = Counter()       # class name -> foreground pixels
    res: Counter = Counter()
    no_mask, corrupt = [], []
    _KEY_LATLON = re.compile(r"^\d+_([0-9.]+)_([0-9.]+)\.(?:jpg|png)$", re.I)
    matched = 0
    for img in images:
        if key == "latlon":
            mm = _KEY_LATLON.match(img.name)
            k = f"{mm.group(1)}_{mm.group(2)}" if mm else None
        else:
            k = img.stem
        if not k:
            no_mask.append(img.name); continue
        found = [p for p in all_masks if p.name.startswith(f"{k}__")]
        if not found:
            no_mask.append(img.name); continue
        matched += 1
        for p in found:
            cls = p.name.split("__")[-1].replace(".png", "")
            cls_counter[cls] += 1
            if do_hist:
                try:
                    from PIL import Image
                    a = np.asarray(Image.open(p).convert("L"))
                    res[Image.open(p).size] += 1
                    px_counter[cls] += int((a > 0).sum())
                except Exception as e:  # noqa: BLE001
                    corrupt.append({"file": p.name, "error": str(e)[:200]})
    # masks whose key matches no image
    keyed = set()
    for img in images:
        mm = _KEY_LATLON.match(img.name)
        if mm:
            keyed.add(f"{mm.group(1)}_{mm.group(2)}")
    orphan = sorted({p.name for p in all_masks if p.name.split("__")[0] not in keyed})
    return {
        "mode": "per_class_binary",
        "n_images": len(images), "n_mask_files": len(all_masks),
        "images_with_masks": matched, "images_without_mask": sorted(no_mask),
        "classes_images": dict(cls_counter.most_common()),
        "classes_fg_pixels": dict(px_counter.most_common()) if do_hist else "skipped",
        "resolutions": {f"{w}x{h}": n for (w, h), n in res.most_common()},
        "orphan_masks_no_image": {"count": len(orphan), "sample": orphan[:20]},
        "corrupt": corrupt,
    }


def _summ(mode, fmt, images, masks, matched, hist, res, un_img, un_mask, corrupt, empty,
          palette=None) -> dict:
    is_rgb = bool(hist) and all(isinstance(k, tuple) for k in hist)
    extra: dict = {}
    if is_rgb and palette:                          # map RGB colours -> class names
        by_name: Counter = Counter()
        unknown: dict = {}
        for color, px in hist.items():
            name = palette.get(color)
            if name:
                by_name[name] += px
            else:
                unknown[f"rgb{color}"] = px
        classes = dict(by_name.most_common())
        if unknown:
            extra["unknown_colors"] = dict(sorted(unknown.items(), key=lambda kv: -kv[1])[:20])
    elif is_rgb:
        classes = {f"rgb{k}": n for k, n in hist.most_common()}
    else:
        classes = {str(k): n for k, n in sorted(hist.items())}
    return {
        "mode": mode, "mask_format": fmt,
        "n_images": len(images), "n_masks": len(masks), "paired": matched,
        "unique_mask_values": len(hist), "class_pixel_counts": classes,
        "resolutions": {f"{w}x{h}": n for (w, h), n in res.most_common()},
        "unmatched_images": {"count": len(un_img), "sample": un_img[:20]},
        "unmatched_masks": {"count": len(un_mask), "sample": list(un_mask)[:20]},
        "corrupt": corrupt, "empty_masks": empty[:50], **extra,
    }


# ------------------------------- runner -------------------------------

def _load_palette(spec) -> dict | None:
    """Palette spec -> {(r,g,b): name}. ``spec`` = path to a YAML/JSON with
    ``classes: [{name, rgb:[r,g,b]}, ...]`` (or an inline list of the same)."""
    if not spec:
        return None
    if isinstance(spec, str):
        text = Path(spec).expanduser().read_text(encoding="utf-8")
        data = json.loads(text) if spec.endswith(".json") else __import__("yaml").safe_load(text)
    else:
        data = spec
    classes = data["classes"] if isinstance(data, dict) else data
    return {tuple(c["rgb"]): c["name"] for c in classes}


def run_source(cfg: dict, do_hist: bool) -> dict:
    name = cfg["name"]
    images = _find(cfg["images"]["root"], cfg["images"].get("glob", "*"))
    limit = int(cfg.get("limit", 0))
    if limit:
        images = images[:limit]
    if cfg.get("mode", "single") == "per_class_binary":
        rec = audit_per_class_binary(images, cfg["masks"]["root"], cfg.get("key", "basename"), do_hist)
        rec["sha256_images"] = _sha256_of_names(images, cfg["images"]["root"])
    else:
        masks = _find(cfg["masks"]["root"], cfg["masks"].get("glob", "*"))
        rec = audit_single(images, masks, cfg["masks"].get("format", "auto"), do_hist,
                           pair_key=cfg["masks"].get("pair_key"),
                           palette=_load_palette(cfg["masks"].get("palette")))
        rec["sha256_images"] = _sha256_of_names(images, cfg["images"]["root"])
        rec["sha256_masks"] = _sha256_of_names(masks, cfg["masks"]["root"])
    rec["name"] = name
    return rec


def _to_md(report: dict) -> str:
    lines = ["# Segmentation data audit", ""]
    for s in report["sources"]:
        lines.append(f"## {s['name']}  (`{s['mode']}`)")
        for k in ("n_images", "n_masks", "n_mask_files", "paired", "images_with_masks",
                  "mask_format", "unique_mask_values"):
            if k in s:
                lines.append(f"- **{k}**: {s[k]}")
        if "resolutions" in s:
            lines.append(f"- **resolutions**: {s['resolutions']}")
        cls = s.get("class_pixel_counts") or s.get("classes_images")
        if cls:
            lines.append(f"- **classes**: {cls}")
        for k in ("unmatched_images", "unmatched_masks", "images_without_mask",
                  "orphan_masks_no_image", "corrupt"):
            if s.get(k):
                lines.append(f"- **{k}**: {s[k] if not isinstance(s[k], list) else len(s[k])}")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML/JSON audit config")
    ap.add_argument("--out-dir", default=None, help="override out_dir from config")
    ap.add_argument("--no-hist", action="store_true", help="skip pixel histograms (fast pass)")
    args = ap.parse_args(argv)

    text = Path(args.config).read_text(encoding="utf-8")
    cfg = json.loads(text) if args.config.endswith(".json") else __import__("yaml").safe_load(text)
    out = Path(args.out_dir or cfg["out_dir"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    report = {"sources": []}
    for src in cfg["sources"]:
        print(f"[audit] {src['name']} ...", flush=True)
        report["sources"].append(run_source(src, do_hist=not args.no_hist))
    (out / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "audit.md").write_text(_to_md(report), encoding="utf-8")
    print(f"[ok] audit.json + audit.md -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
