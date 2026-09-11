"""DINOv2 / DINOv3 dense-feature extractors + RandomProjector (vendored).

Byte-faithful port of the four symbols the extractor uses from RevisitAnything's
1173-line ``utilities.py`` — ``DinoV2ExtractFeatures``, ``DinoV3ExtractFeatures``,
``build_dino_extractor``, ``RandomProjector`` — with none of the unused faiss / sklearn /
transformers / matplotlib baggage. Only torch (+ torch.hub → torchvision) is required.

torch is imported at module top, so import this only where the model is actually used
(the extractor), not from unit tests.
"""
from __future__ import annotations

import os
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]
_DINO_V3_MODELS = Literal[
    "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16",
    "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16",
]


def _local_hub_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hub")
    return d if os.path.isdir(d) else None


class DinoV2ExtractFeatures:
    """Extract features from an intermediate layer of a DINOv2 backbone (torch.hub)."""

    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int,
                 facet: _DINO_FACETS = "token", use_cls=False,
                 norm_descs=True, device: str = "cpu") -> None:
        self.vit_type: str = dino_model
        _hub = _local_hub_dir()
        if _hub:
            torch.hub.set_dir(_hub)
        self.dino_model: nn.Module = torch.hub.load('facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].register_forward_hook(
                self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].attn.qkv.register_forward_hook(
                self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self._hook_out = None

    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2 * d_len]
                else:
                    res = res[:, :, 2 * d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None
        return res

    def __del__(self):
        if getattr(self, "fh_handle", None) is not None:
            self.fh_handle.remove()


class DinoV3ExtractFeatures:
    """Dense patch features from a DINOv3 backbone (torch.hub, patch-16, token facet)."""

    patch_size = 16

    def __init__(self, dino_model: _DINO_V3_MODELS = "dinov3_vitl16",
                 layer: int | None = None, facet: _DINO_FACETS = "token",
                 device: str = "cpu", norm_descs: bool = True,
                 weights: str | None = None) -> None:
        self.vit_type = dino_model
        if facet not in (None, "token"):
            print(f"[dinov3] facet={facet!r} not supported — using 'token' features.")
        self.facet = "token"
        _hub = _local_hub_dir()
        if _hub:
            torch.hub.set_dir(_hub)
        load_kwargs = {} if weights is None else {"weights": weights}
        self.dino_model = torch.hub.load("facebookresearch/dinov3", dino_model, **load_kwargs)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)

        n_blocks = len(self.dino_model.blocks)
        if layer is None:
            self.layer = n_blocks - 1
        elif layer < 0:
            self.layer = n_blocks + layer
        else:
            self.layer = min(layer, n_blocks - 1)
        self.embed_dim = getattr(self.dino_model, "embed_dim", None)
        self.norm_descs = norm_descs

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.dino_model.get_intermediate_layers(
                img, n=[self.layer], reshape=False,
                return_class_token=False, norm=False,
            )
            res = feats[0]
            if self.norm_descs:
                res = F.normalize(res, dim=-1)
        return res


def build_dino_extractor(backbone: str, *, layer: int | None = None,
                         facet: str = "value", device: str = "cpu",
                         norm_descs: bool = False, weights: str | None = None):
    """Return a DINOv2 or DINOv3 extractor by name; both expose ``.patch_size``."""
    if backbone.startswith("dinov2"):
        ext = DinoV2ExtractFeatures(
            backbone, 31 if layer is None else layer, facet,
            device=device, norm_descs=norm_descs,
        )
        ext.patch_size = 14
        return ext
    if backbone.startswith("dinov3"):
        return DinoV3ExtractFeatures(
            backbone, layer=layer, facet=facet,
            device=device, norm_descs=norm_descs, weights=weights,
        )
    raise ValueError(f"Unknown DINO backbone: {backbone!r} "
                     "(expected a 'dinov2_*' or 'dinov3_*' name)")


class RandomProjector:
    """Deterministic Johnson–Lindenstrauss projection to a chosen descriptor dim.

    Reduces DINO patch features ``(..., D_in)`` → ``(..., out_dim)`` with a fixed Gaussian
    matrix built on CPU from ``seed`` (reproducible), scaled by ``1/sqrt(out_dim)``. Map
    and query extraction MUST use the same (backbone, out_dim, seed). ``out_dim`` None or
    >= D_in is a pass-through.
    """

    def __init__(self, out_dim: int | None, seed: int = 0) -> None:
        self.out_dim = out_dim
        self.seed = int(seed)
        self.in_dim: int | None = None
        self._P: torch.Tensor | None = None
        self._warned = False

    @property
    def active(self) -> bool:
        return self._P is not None

    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        if self.out_dim is None:
            return feat
        d_in = feat.shape[-1]
        if self.out_dim >= d_in:
            if not self._warned:
                print(f"[proj] --desc-dim {self.out_dim} >= native {d_in}; "
                      "keeping native dimension (no projection).")
                self._warned = True
            self.out_dim = None
            return feat
        if self._P is None:
            g = torch.Generator(device="cpu").manual_seed(self.seed)
            P = torch.randn(d_in, self.out_dim, generator=g) / (self.out_dim ** 0.5)
            self.in_dim = d_in
            self._P = P.to(feat.device, feat.dtype)
        return feat @ self._P

    def h5_attrs(self) -> dict:
        if not self.active:
            return {"projection": "none"}
        return {
            "projection": "random_gaussian_jl",
            "projection_seed": self.seed,
            "projection_in_dim": int(self.in_dim),
            "projection_out_dim": int(self.out_dim),
        }
