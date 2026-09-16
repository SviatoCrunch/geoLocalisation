# VENDORED verbatim from siam_model_stage3/features/uav_augment.py (that pkg is a separate repo,
# absent on the geoLocalisation server checkout). Do not edit here; sync from source if it changes.
"""Train-only UAV image augmentation — a faithful port of the successful Stage-2 recipe
(`siam_model_stage2/train.py:_augment_uav_image` + the rotation in `drone_tokens`), adapted to
Stage-3's precomputed-sky-mask pipeline.

SOURCE OF TRUTH = real siam_model_stage2/train.py runtime code (NOT the task-spec YAML — that was an
assumption to reconcile against the actual Stage-2 code, and the literal code wins). The LITERAL
Stage-2 recipe is reproduced by ``preset: stage2_exact`` (these are the ``AugConfig`` DEFAULTS):

  GEOMETRIC (applied to RGB *and*, synchronously, to the sky mask with nearest-neighbour):
    1. scale        p=0.80  s=exp(U(ln0.75, ln1.35)) log-uniform; BILINEAR; s>1 center-crop,
                            s<1 edge-pad (replicate).
    2. translate    p=0.50  dx,dy=randint(±0.08·W/H); np.roll (circular wrap)  ⚠ Stage-2 behaviour
                            (kept for parity; opt-in fix: translate_mode=shift_pad/affine_fill, no wrap).
    3. perspective  p=0.25  4 corners U(±0.04); PIL PERSPECTIVE; SKIPPED if scale was applied
                            (effective marginal P = (1-0.80)·0.25 = 0.05).
    4. rotation     p=1.00  angle=U(±12°); PIL rotate expand=False (separate op in Stage-2).
  PHOTOMETRIC (RGB ONLY — never the mask):
    5. brightness   p=0.60  ×(1+U(±0.15))
    6. contrast     p=0.60  ×U(0.80,1.20)
    7. gamma        p=0.40  g=U(0.80,1.25)
    8. jpeg         p=0.50  quality=randint[35,95] PIL round-trip
    9. blur         p=0.25  GaussianBlur radius=U(0.2,1.5)
   10. noise        p=0.25  additive N(0, U(0,0.025)), clip[0,1]
  flips: NONE (Stage-2 had none — fail-closed if enabled).

``preset: stage3_strong`` is a deliberately STRONGER experimental recipe (NOT Stage-2, declared in
``_PRESET_OVERRIDES``): rot ±30° p1.0, scale 0.60-1.35→1.60 p0.70, translate 0.15 p1.0 via the SAFE
``affine_fill`` mode (shift+fill, NO wraparound), perspective ±0.08 p1.0 (effective 0.30), jpeg 20-85
p1.0, blur/noise p0.50; brightness/contrast/gamma stay as Stage-2. Same op order and mutual exclusion.

Divergences from Stage-2 (documented, not silent):
  * Stage-2 re-SEGMENTS sky online on the augmented image and DROPS sky tokens. Stage-3 keeps its
    precomputed-mask FILL pipeline; here the mask is TRANSFORMED synchronously (nearest, binary) and
    out-of-frame regions introduced by perspective/rotation are marked sky=True so the downstream
    sky-fill replaces them with the ground mean (no black patches reach DINO).
  * ``no flips`` (Stage-2 had none) — omitted on purpose (oblique UAV).

Geometric params are sampled ONCE per sample and applied identically to RGB and mask. Photometric
transforms touch RGB only. Augmentation is TRAIN-ONLY (val/test call the model with the cached
un-augmented tokens)."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field, asdict

import numpy as np

VALID_PRESETS = ("stage2_exact", "stage3_strong")
VALID_TRANSLATE_MODES = ("roll", "shift_pad", "affine_fill")
# provenance marker: when the semantics of ``stage2_exact`` were fixed to the LITERAL Stage-2 recipe.
PRESET_SEMANTICS = "v2_literal_stage2"

# ``stage2_exact`` is the SOURCE-OF-TRUTH preset: it reproduces the LITERAL siam_model_stage2/train.py
# runtime (``_augment_uav_image`` + the rotation in ``drone_tokens``) — that is the dataclass default
# below, so a bare ``stage2_exact`` == real Stage-2. ``stage3_strong`` is a deliberately stronger
# experimental recipe (NOT Stage-2): it also switches translation to the safe ``affine_fill`` mode
# (no wraparound) instead of Stage-2's ``np.roll``. Only the values that DIFFER from the literal
# baseline are listed; order, mutual exclusion, and photometric ops are identical to Stage-2.
_PRESET_OVERRIDES = {
    "stage2_exact": {},                    # literal Stage-2 == the dataclass defaults
    "stage3_strong": {                     # stronger experimental recipe (declared config)
        "rot_max_deg": 30.0, "rot_prob": 1.0,
        "scale_min": 0.60, "scale_max": 1.60, "scale_prob": 0.70,
        "translate_frac": 0.15, "translate_prob": 1.0, "translate_mode": "affine_fill",
        "persp_scale": 0.08, "persp_prob": 1.0, "persp_exclusive_with_scale": True,
        "jpeg_qmin": 20, "jpeg_qmax": 85, "jpeg_prob": 1.0,
        "blur_prob": 0.50, "noise_prob": 0.50,
        # brightness/contrast/gamma stay as Stage-2 (0.60/±0.15, 0.60/0.80-1.20, 0.40/0.80-1.25)
    },
}


@dataclass
class AugConfig:
    # defaults == LITERAL siam_model_stage2/train.py (the ``stage2_exact`` preset)
    enabled: bool = False
    preset: str = "stage2_exact"
    rot_max_deg: float = 12.0                     # --rot-max-deg 12
    rot_prob: float = 1.0                         # rng.uniform(-12,12) always sampled
    scale_min: float = 0.75
    scale_max: float = 1.35
    scale_prob: float = 0.80
    scale_distribution: str = "log_uniform"       # only log_uniform is implemented (Stage-2)
    translate_frac: float = 0.08
    translate_prob: float = 0.50
    translate_mode: str = "roll"                  # roll = literal Stage-2 (np.roll wraparound)
    persp_scale: float = 0.04
    persp_prob: float = 0.25
    persp_exclusive_with_scale: bool = True
    bright_prob: float = 0.60
    bright_delta: float = 0.15
    contrast_prob: float = 0.60
    contrast_min: float = 0.80
    contrast_max: float = 1.20
    gamma_prob: float = 0.40
    gamma_min: float = 0.80
    gamma_max: float = 1.25
    jpeg_prob: float = 0.50
    jpeg_qmin: int = 35
    jpeg_qmax: int = 95
    blur_prob: float = 0.25
    blur_rmin: float = 0.2
    blur_rmax: float = 1.5
    noise_prob: float = 0.25
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.025
    flips_enabled: bool = False                   # Stage-2 had NO flips (fail-closed if enabled)
    max_resample: int = 3
    min_valid_fraction: float = 0.05
    modified_from_preset: bool = False           # provenance: preset was overridden

    # ── build + validate ──────────────────────────────────────────────────────────────
    @classmethod
    def from_cfg(cls, aug_block: dict | None, overrides: dict | None = None) -> "AugConfig":
        """Build from a config ``augmentation`` block (+ optional CLI overrides). ``preset`` seeds the
        Stage-2 values; explicit keys override them and set ``modified_from_preset=True``."""
        aug_block = dict(aug_block or {})
        preset = aug_block.get("preset", "stage2_exact")
        if preset not in VALID_PRESETS:
            raise ValueError(f"augmentation.preset={preset!r} unknown; expected {VALID_PRESETS}.")
        cfg = cls(enabled=bool(aug_block.get("enabled", False)), preset=preset)  # stage2_exact defaults
        for k, v in _PRESET_OVERRIDES[preset].items():       # seed the chosen preset's values
            setattr(cfg, k, v)
        modified = False
        merged = {**_flatten_block(aug_block), **(overrides or {})}
        for k, v in merged.items():
            if k in ("enabled", "preset") or v is None:
                continue
            if not hasattr(cfg, k):
                raise ValueError(f"augmentation: unknown parameter {k!r}.")
            if getattr(cfg, k) != v:            # only a value that DIFFERS from the preset counts
                modified = True
            setattr(cfg, k, v)
        cfg.modified_from_preset = modified
        cfg.validate()
        return cfg

    def validate(self) -> "AugConfig":
        if self.preset not in VALID_PRESETS:
            raise ValueError(f"augmentation.preset={self.preset!r} unknown; expected {VALID_PRESETS}.")
        if self.translate_mode not in VALID_TRANSLATE_MODES:
            raise ValueError(f"augmentation.translate_mode={self.translate_mode!r}; "
                             f"expected {VALID_TRANSLATE_MODES}.")
        if self.scale_min <= 0:
            raise ValueError(f"scale_min must be > 0 (got {self.scale_min}).")
        if self.scale_max < self.scale_min:
            raise ValueError(f"scale_max {self.scale_max} < scale_min {self.scale_min}.")
        for name in ("rot_prob", "scale_prob", "translate_prob", "persp_prob", "bright_prob",
                     "contrast_prob", "gamma_prob", "jpeg_prob", "blur_prob", "noise_prob"):
            p = float(getattr(self, name))
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name}={p} out of [0,1].")
        if not (1 <= self.jpeg_qmin <= 100 and 1 <= self.jpeg_qmax <= 100):
            raise ValueError(f"jpeg quality must be in [1,100] (got {self.jpeg_qmin}..{self.jpeg_qmax}).")
        if self.jpeg_qmax < self.jpeg_qmin:
            raise ValueError(f"jpeg_qmax {self.jpeg_qmax} < jpeg_qmin {self.jpeg_qmin}.")
        for name in ("rot_max_deg", "translate_frac", "persp_scale", "noise_sigma_max",
                     "bright_delta"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0 (got {getattr(self, name)}).")
        if self.contrast_max < self.contrast_min or self.gamma_max < self.gamma_min:
            raise ValueError("contrast/gamma max < min.")
        if self.blur_rmax < self.blur_rmin:
            raise ValueError("blur_rmax < blur_rmin.")
        if self.noise_sigma_min < 0 or self.noise_sigma_max < self.noise_sigma_min:
            raise ValueError(f"noise sigma range invalid ({self.noise_sigma_min}..{self.noise_sigma_max}).")
        if self.scale_distribution != "log_uniform":
            raise ValueError(f"scale.distribution={self.scale_distribution!r} unsupported "
                             "(only 'log_uniform' is implemented — matches Stage-2).")
        if bool(self.flips_enabled):
            raise ValueError("flips are NOT supported (Stage-2 had none); set flips.enabled=false.")
        return self

    def perspective_effective_probability(self) -> float:
        """Marginal P(perspective): perspective is (optionally) mutually exclusive with scale, so it can
        only fire when scale did NOT. Effective = (1 - P(scale))·P(persp|¬scale) when exclusive."""
        cond = float(self.persp_prob)
        if self.persp_exclusive_with_scale:
            return (1.0 - float(self.scale_prob)) * cond
        return cond

    def summary_lines(self) -> list[str]:
        return [
            f"[aug] preset={self.preset} enabled={self.enabled} semantics={PRESET_SEMANTICS}"
            + ("  (MODIFIED from preset)" if self.modified_from_preset else ""),
            f"[aug] rotation=±{self.rot_max_deg}deg p={self.rot_prob}",
            f"[aug] scale=[{self.scale_min},{self.scale_max}] p={self.scale_prob} ({self.scale_distribution})",
            f"[aug] translate=frac{self.translate_frac} p={self.translate_prob} mode={self.translate_mode}",
            f"[aug] perspective=±{self.persp_scale} "
            f"perspective_conditional_probability={self.persp_prob} "
            f"perspective_effective_probability={self.perspective_effective_probability():.4f} "
            f"exclusive_with_scale={self.persp_exclusive_with_scale}",
            f"[aug] jpeg=q[{self.jpeg_qmin},{self.jpeg_qmax}] p={self.jpeg_prob}",
            f"[aug] blur=r[{self.blur_rmin},{self.blur_rmax}] p={self.blur_prob}",
            f"[aug] noise=sigma[{self.noise_sigma_min},{self.noise_sigma_max}] p={self.noise_prob}",
            f"[aug] photometric: brightness p={self.bright_prob} contrast p={self.contrast_prob} "
            f"gamma p={self.gamma_prob}  flips={self.flips_enabled}",
            "[aug] synchronized_sky_mask=true  applied_before_dino=true  train_only=true",
        ]

    def fingerprint(self) -> str:
        import hashlib
        import json
        return "aug:" + hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:32]


def _flatten_block(block: dict) -> dict:
    """Accept either a flat block (rot_max_deg=…) or the nested YAML form (rotation:{max_degrees,…})."""
    if not block:
        return {}
    nested = {
        ("rotation", "max_degrees"): "rot_max_deg", ("rotation", "probability"): "rot_prob",
        ("scale", "min"): "scale_min", ("scale", "max"): "scale_max", ("scale", "probability"): "scale_prob",
        ("scale", "distribution"): "scale_distribution",
        ("translate", "max_fraction"): "translate_frac", ("translate", "probability"): "translate_prob",
        ("translate", "mode"): "translate_mode",
        ("perspective", "distortion_scale"): "persp_scale", ("perspective", "probability"): "persp_prob",
        ("perspective", "exclusive_with_scale"): "persp_exclusive_with_scale",
        ("brightness", "probability"): "bright_prob", ("brightness", "delta"): "bright_delta",
        ("contrast", "probability"): "contrast_prob", ("contrast", "min"): "contrast_min",
        ("contrast", "max"): "contrast_max",
        ("gamma", "probability"): "gamma_prob", ("gamma", "min"): "gamma_min", ("gamma", "max"): "gamma_max",
        ("jpeg", "quality_min"): "jpeg_qmin", ("jpeg", "quality_max"): "jpeg_qmax",
        ("jpeg", "probability"): "jpeg_prob",
        ("blur", "probability"): "blur_prob", ("blur", "radius_min"): "blur_rmin",
        ("blur", "radius_max"): "blur_rmax",
        ("noise", "probability"): "noise_prob", ("noise", "sigma_min"): "noise_sigma_min",
        ("noise", "sigma_max"): "noise_sigma_max",
        ("flips", "enabled"): "flips_enabled",
    }
    out = {}
    for k, v in block.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat = nested.get((k, kk))
                if flat:
                    out[flat] = vv
        elif k not in ("enabled", "preset"):
            out[k] = v
    return out


@dataclass
class AugStats:
    augmentation_samples: int = 0
    augmentation_resamples: int = 0
    augmentation_fallbacks: int = 0
    missing_sky_masks: int = 0
    _vf_before: list = field(default_factory=list)
    _vf_after: list = field(default_factory=list)

    def as_dict(self) -> dict:
        mb = float(np.mean(self._vf_before)) if self._vf_before else None
        ma = float(np.mean(self._vf_after)) if self._vf_after else None
        return {"augmentation_samples": self.augmentation_samples,
                "augmentation_resamples": self.augmentation_resamples,
                "augmentation_fallbacks": self.augmentation_fallbacks,
                "missing_sky_masks": self.missing_sky_masks,
                "mean_valid_fraction_before": mb, "mean_valid_fraction_after": ma}


# ── parameter sampling (ONCE per sample) ───────────────────────────────────────────────
def sample_params(cfg: AugConfig, rng, W: int, H: int) -> dict:
    p: dict = {"W": W, "H": H}
    p["do_scale"] = rng.rand() < cfg.scale_prob
    if p["do_scale"]:
        p["scale"] = math.exp(rng.uniform(math.log(cfg.scale_min), math.log(cfg.scale_max)))
    p["do_translate"] = rng.rand() < cfg.translate_prob
    if p["do_translate"]:
        mdx = max(1, int(W * cfg.translate_frac)); mdy = max(1, int(H * cfg.translate_frac))
        p["dx"] = int(rng.randint(-mdx, mdx + 1)); p["dy"] = int(rng.randint(-mdy, mdy + 1))
    # perspective is mutually exclusive with scale (Stage-2)
    p["do_persp"] = (not (cfg.persp_exclusive_with_scale and p["do_scale"])) and \
        (rng.rand() < cfg.persp_prob)
    if p["do_persp"]:
        f = cfg.persp_scale
        p["persp_sh"] = [(rng.uniform(-f, f), rng.uniform(-f, f)) for _ in range(4)]
    p["do_rot"] = rng.rand() < cfg.rot_prob
    p["angle"] = float(rng.uniform(-cfg.rot_max_deg, cfg.rot_max_deg)) if p["do_rot"] else 0.0
    # photometric
    p["do_bright"] = rng.rand() < cfg.bright_prob
    p["bright"] = 1.0 + rng.uniform(-cfg.bright_delta, cfg.bright_delta)
    p["do_contrast"] = rng.rand() < cfg.contrast_prob
    p["contrast"] = rng.uniform(cfg.contrast_min, cfg.contrast_max)
    p["do_gamma"] = rng.rand() < cfg.gamma_prob
    p["gamma"] = rng.uniform(cfg.gamma_min, cfg.gamma_max)
    p["do_jpeg"] = rng.rand() < cfg.jpeg_prob
    p["jpeg_q"] = int(rng.randint(cfg.jpeg_qmin, cfg.jpeg_qmax + 1))
    p["do_blur"] = rng.rand() < cfg.blur_prob
    p["blur_r"] = rng.uniform(cfg.blur_rmin, cfg.blur_rmax)
    p["do_noise"] = rng.rand() < cfg.noise_prob
    p["noise_sigma"] = rng.uniform(cfg.noise_sigma_min, cfg.noise_sigma_max)
    return p


# ── geometric (RGB via BILINEAR, mask via NEAREST — SAME params) ────────────────────────
def _apply_geometric(pil, p: dict, cfg: AugConfig, *, is_mask: bool):
    from PIL import Image as _Img
    W0, H0 = cfg_wh = (p["W"], p["H"])
    resample = _Img.Resampling.NEAREST if is_mask else _Img.Resampling.BILINEAR
    fill = 255 if is_mask else 0                       # mask out-of-frame → sky(255); rgb → black
    # 1. scale
    if p["do_scale"]:
        s = p["scale"]; nw, nh = max(1, round(W0 * s)), max(1, round(H0 * s))
        pil = pil.resize((nw, nh), resample)
        if s > 1.0:
            x0, y0 = (nw - W0) // 2, (nh - H0) // 2
            pil = pil.crop((x0, y0, x0 + W0, y0 + H0))
        else:
            px, py = (W0 - nw) // 2, (H0 - nh) // 2
            arr = np.array(pil)
            pad = ((py, H0 - nh - py), (px, W0 - nw - px)) + (((0, 0),) if arr.ndim == 3 else ())
            pil = _Img.fromarray(np.pad(arr, pad, mode="edge"))
    # 2. translate
    if p["do_translate"]:
        dx, dy = p["dx"], p["dy"]; arr = np.array(pil)
        if cfg.translate_mode == "roll":               # Stage-2 exact (circular wrap — NO fill)
            if dy: arr = np.roll(arr, dy, axis=0)
            if dx: arr = np.roll(arr, dx, axis=1)
        elif cfg.translate_mode == "affine_fill":      # safe shift: NO wraparound; vacated → fill
            H, W = arr.shape[:2]
            out = np.full_like(arr, fill)              # mask→255(sky) / rgb→0(black→sky-filled later)
            ys0, ys1 = max(0, dy), min(H, H + dy); sy0, sy1 = max(0, -dy), min(H, H - dy)
            xs0, xs1 = max(0, dx), min(W, W + dx); sx0, sx1 = max(0, -dx), min(W, W - dx)
            if ys1 > ys0 and xs1 > xs0:
                out[ys0:ys1, xs0:xs1] = arr[sy0:sy1, sx0:sx1]
            arr = out
        else:                                          # shift_pad (opt-in fix): shift + edge-pad
            arr = np.roll(arr, (dy, dx), axis=(0, 1))
            if dy > 0: arr[:dy] = arr[dy:dy + 1]
            elif dy < 0: arr[dy:] = arr[dy - 1:dy]
            if dx > 0: arr[:, :dx] = arr[:, dx:dx + 1]
            elif dx < 0: arr[:, dx:] = arr[:, dx - 1:dx]
        pil = _Img.fromarray(arr)
    # 3. perspective (skipped if scale applied)
    if p["do_persp"]:
        W, H = pil.size; sh = p["persp_sh"]
        src = [(0, 0), (W, 0), (W, H), (0, H)]
        dst = [(W * sh[0][0], H * sh[0][1]), (W + W * sh[1][0], H * sh[1][1]),
               (W + W * sh[2][0], H + H * sh[2][1]), (W * sh[3][0], H + H * sh[3][1])]
        pil = pil.transform((W, H), _Img.Transform.PERSPECTIVE,
                            _find_perspective_coeffs(src, dst), resample, fillcolor=fill)
    # 4. rotation (separate op, after augment in Stage-2)
    if p["do_rot"] and abs(p["angle"]) > 1e-3:
        pil = pil.rotate(p["angle"], resample=resample, expand=False, fillcolor=fill)
    return pil


def _find_perspective_coeffs(pa, pb):
    """8-coeff perspective transform for PIL (verbatim from Stage-2)."""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix += [[p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]],
                   [0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]]]
    A = np.array(matrix, dtype=float)
    b = np.array([c for pt in pb for c in pt], dtype=float)
    return np.linalg.lstsq(A, b, rcond=None)[0].tolist()


def augment_frame(rgb: np.ndarray, sky_mask: np.ndarray | None, cfg: AugConfig, rng,
                  stats: AugStats | None = None):
    """Augment one UAV frame. ``rgb`` (H,W,3) uint8; ``sky_mask`` (H,W) bool True==sky or None.

    Geometric params are sampled once and applied to BOTH rgb and mask (mask nearest, kept binary);
    photometric to rgb only. Validity-checked with limited resampling → safe fallback (original) on
    failure. Returns (aug_rgb uint8, aug_mask bool|None, params, valid). Deterministic given ``rng``."""
    from PIL import Image as _Img
    H0, W0 = rgb.shape[:2]
    if stats is not None:
        stats.augmentation_samples += 1
        if sky_mask is None:
            stats.missing_sky_masks += 1
        stats._vf_before.append(1.0 if sky_mask is None else float((~sky_mask).mean()))
    if not cfg.enabled:
        return rgb, sky_mask, {}, True

    last = (rgb, sky_mask, {})
    for attempt in range(max(1, cfg.max_resample)):
        p = sample_params(cfg, rng, W0, H0)
        rgb_pil = _apply_geometric(_Img.fromarray(rgb), p, cfg, is_mask=False)
        rgb_pil = _apply_photometric_rng(rgb_pil, p, rng)
        aug_rgb = np.asarray(rgb_pil)
        aug_mask = None
        if sky_mask is not None:
            m_pil = _Img.fromarray((sky_mask.astype(np.uint8) * 255))
            m_pil = _apply_geometric(m_pil, p, cfg, is_mask=True)
            aug_mask = np.asarray(m_pil) > 127
        valid_frac = 1.0 if aug_mask is None else float((~aug_mask).mean())
        ok = (aug_rgb.shape == rgb.shape and aug_rgb.dtype == np.uint8
              and np.isfinite(aug_rgb).all() and aug_rgb.max() > 0
              and (aug_mask is None or aug_mask.shape == sky_mask.shape)
              and valid_frac >= cfg.min_valid_fraction)
        if ok:
            if stats is not None:
                stats._vf_after.append(valid_frac)
                if attempt > 0:
                    stats.augmentation_resamples += attempt
            return aug_rgb, aug_mask, p, True
        last = (aug_rgb, aug_mask, p)
    # fallback: original, un-augmented (never hang)
    if stats is not None:
        stats.augmentation_resamples += max(0, cfg.max_resample - 1)
        stats.augmentation_fallbacks += 1
        stats._vf_after.append(1.0 if sky_mask is None else float((~sky_mask).mean()))
    return rgb, sky_mask, {"fallback": True}, False


def _apply_photometric_rng(pil, p: dict, rng):
    """Photometric with the sample's own rng for the noise draw (deterministic)."""
    from PIL import Image as _Img, ImageEnhance, ImageFilter
    if p["do_bright"]:
        pil = ImageEnhance.Brightness(pil).enhance(p["bright"])
    if p["do_contrast"]:
        pil = ImageEnhance.Contrast(pil).enhance(p["contrast"])
    if p["do_gamma"]:
        arr = np.array(pil).astype(np.float32) / 255.0
        pil = _Img.fromarray((np.power(arr.clip(0, 1), p["gamma"]) * 255).astype(np.uint8))
    if p["do_jpeg"]:
        buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=p["jpeg_q"]); buf.seek(0)
        pil = _Img.open(buf).copy()
    if p["do_blur"]:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=p["blur_r"]))
    if p["do_noise"] and p["noise_sigma"] > 0:
        arr = np.array(pil).astype(np.float32) / 255.0
        arr = (arr + rng.normal(0, p["noise_sigma"], arr.shape)).clip(0, 1)
        pil = _Img.fromarray((arr * 255).astype(np.uint8))
    return pil


# ── deterministic RNG derivation (§10: global seed × epoch × worker → reproducible, non-colliding) ──
_AUG_SEED_OFFSET = 0x5A6E                 # separates the augmentation RNG stream from the sampler's rng


def derive_seed(base_seed: int, epoch: int = 0, worker_id: int = 0) -> int:
    """A reproducible per-(seed, epoch, worker) 32-bit seed. Same inputs → same stream (reproducible
    run); different epoch → different augmentation each epoch; different worker → non-colliding streams
    (so parallel workers never replay the SAME augmentation sequence). Deterministic, no wall-clock."""
    import hashlib
    h = hashlib.sha256(f"{int(base_seed)}:{int(epoch)}:{int(worker_id)}:{_AUG_SEED_OFFSET}".encode())
    return int.from_bytes(h.digest()[:4], "big")


def make_aug_rng(base_seed: int, epoch: int = 0, worker_id: int = 0):
    """A ``numpy.random.RandomState`` seeded via :func:`derive_seed` (the canonical augmentation RNG)."""
    return np.random.RandomState(derive_seed(base_seed, epoch, worker_id))


def resolve_aug_block(cfg_aug: dict | None, cli_preset=None, cli_enabled=None) -> dict:
    """Merge a config ``augmentation`` block with CLI ``--aug-preset`` / ``--aug-enabled`` overrides.

    IMPORTANT: a YAML block documents the DEFAULT preset's per-parameter values. If the CLI switches to
    a DIFFERENT preset, those per-parameter YAML keys belong to the OTHER preset and MUST NOT clobber
    the new preset — so they are dropped (only ``enabled``/``preset`` survive) and the new preset's
    canonical values are used (``--aug-override`` still applies on top, in the caller)."""
    block = dict(cfg_aug or {})
    yaml_preset = block.get("preset", "stage2_exact")
    if cli_preset is not None and cli_preset != yaml_preset:
        block = {"enabled": block.get("enabled", False), "preset": cli_preset}   # drop other preset's params
    elif cli_preset is not None:
        block["preset"] = cli_preset
    if cli_enabled is not None:
        block["enabled"] = bool(cli_enabled)
    return block
