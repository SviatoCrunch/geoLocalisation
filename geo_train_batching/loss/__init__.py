"""Training losses (model-agnostic): multi-positive CE, full-gallery NLL, InfoNCE."""
from __future__ import annotations

from .full_gallery import (dense_full_gallery_nll, streaming_full_gallery_nll,
                           streaming_soft_relevance_ce, validate_masks)
from .infonce import symmetric_infonce, LOGIT_SCALE_DEFAULT
from .multipositive import (TAU_LOSS_DEFAULT, masked_logsumexp, multipositive_ce,
                            symmetric_multipositive_ce)

__all__ = [
    "TAU_LOSS_DEFAULT", "masked_logsumexp", "multipositive_ce", "symmetric_multipositive_ce",
    "dense_full_gallery_nll", "streaming_full_gallery_nll", "streaming_soft_relevance_ce",
    "validate_masks", "symmetric_infonce", "LOGIT_SCALE_DEFAULT",
]
