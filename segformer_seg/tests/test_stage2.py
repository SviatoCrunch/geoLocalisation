"""Stage-2 units: class weights, confusion-matrix IoU, combined loss, dataset transforms."""
import numpy as np
import torch

from segformer_seg.class_weights import compute_weights
from segformer_seg.config import CLASSES, IGNORE_INDEX, NUM_CLASSES
from segformer_seg.losses import CombinedLoss, DiceLoss
from segformer_seg.metrics import ConfusionMatrix


def test_median_freq_weights_lift_rare_and_clip():
    # background huge, bridge tiny -> bridge weight high, background low, clipped at 15
    # order: background, bridge, railway, road, sky, water
    counts = [68_000_000, 56_000, 1_130_000, 2_920_000, 17_000_000, 320_000]
    w = compute_weights(counts, scheme="median", clip=15.0)
    assert w[CLASSES.index("background")] < 0.1
    assert w[CLASSES.index("bridge")] == 15.0           # clipped (rarest)
    assert w[CLASSES.index("water")] > w[CLASSES.index("road")]
    assert len(w) == NUM_CLASSES


def test_weights_ignore_background():
    counts = [10] + [1] * (NUM_CLASSES - 1)
    w = compute_weights(counts, ignore_background=True)
    assert w[0] == 0.0


def test_confusion_iou_perfect_and_ignore():
    cm = ConfusionMatrix()
    gt = np.array([[0, 1], [2, IGNORE_INDEX]])
    cm.update(gt, gt)                                   # perfect where not ignored
    s = cm.summary()
    assert abs(s["per_class_iou"]["background"] - 1.0) < 1e-9
    assert s["per_class_iou"]["sky"] is None            # class absent -> None, not counted
    assert 0.99 < s["mIoU"] <= 1.0


def test_combined_loss_runs_and_ignores_padding():
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_CLASSES, 8, 8, requires_grad=True)
    target = torch.randint(0, NUM_CLASSES, (2, 8, 8))
    target[:, 0, :] = IGNORE_INDEX                      # a padded row
    loss = CombinedLoss(class_weights=[1.0] * NUM_CLASSES, dice_w=1.0)(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None


def test_dice_perfect_prediction_near_zero():
    target = torch.randint(0, NUM_CLASSES, (1, 6, 6))
    onehot = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    logits = (onehot - 0.5) * 50                        # confident correct logits
    assert DiceLoss()(logits, target).item() < 0.05
