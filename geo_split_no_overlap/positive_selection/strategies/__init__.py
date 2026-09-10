"""Built-in strategy implementations. Importing this package does NOT construct any
strategy — only classes/factories are defined. Registration happens in
``positive_selection.registry``.
"""
from __future__ import annotations

from .contains_point import ContainsPointSelector
from .current_rule import CurrentRuleSelector
from .pyramid_top_iou_250 import PyramidTopIoU250Selector
from .tile_iou_1000 import TileIoU1000Selector

__all__ = ["CurrentRuleSelector", "ContainsPointSelector",
           "TileIoU1000Selector", "PyramidTopIoU250Selector"]
