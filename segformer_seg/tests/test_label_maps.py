"""Label-map composition: paint order, background, name reconciliation, PNG resolution."""
import numpy as np
from PIL import Image

from segformer_seg.config import CLASS_TO_ID, canonical
from segformer_seg.label_maps import compose_from_coco, compose_from_pngs


def test_canonical_aliases_and_drops():
    assert canonical("Railways") == "railway"
    assert canonical("road") == "road"
    assert canonical("Building") is None   # server-only class is dropped
    assert canonical("Target") is None
    assert canonical("bridge") == "bridge"


def test_png_paint_order_thin_over_large(tmp_path):
    # sky covers the whole frame; railway a thin band that must win the overlap.
    h, w = 8, 8
    sky = np.full((h, w), 255, np.uint8)
    rail = np.zeros((h, w), np.uint8); rail[3:5, :] = 255
    Image.fromarray(sky).save(tmp_path / "s.png")
    Image.fromarray(rail).save(tmp_path / "r.png")
    lab = compose_from_pngs([("sky", str(tmp_path / "s.png")),
                             ("railway", str(tmp_path / "r.png"))], (h, w))
    assert (lab[3:5, :] == CLASS_TO_ID["railway"]).all()      # railway on top
    assert (lab[0, :] == CLASS_TO_ID["sky"]).all()            # sky elsewhere
    assert CLASS_TO_ID["background"] not in np.unique(lab)    # fully covered


def test_png_background_where_unlabelled(tmp_path):
    h, w = 6, 6
    road = np.zeros((h, w), np.uint8); road[:, :2] = 255
    Image.fromarray(road).save(tmp_path / "road.png")
    lab = compose_from_pngs([("road", str(tmp_path / "road.png"))], (h, w))
    assert (lab[:, 2:] == 0).all()                            # background
    assert (lab[:, :2] == CLASS_TO_ID["road"]).all()


def test_coco_polygon_rasterised_and_dropped_classes():
    h, w = 10, 10
    cats = {1: "road", 2: "Building"}   # Building -> dropped
    anns = [
        {"category_id": 1, "segmentation": [[0, 0, 9, 0, 9, 4, 0, 4]]},  # top band = road
        {"category_id": 2, "segmentation": [[0, 5, 9, 5, 9, 9, 0, 9]]},  # bottom = dropped
    ]
    lab = compose_from_coco(anns, cats, (h, w))
    assert (lab[0:4, :] == CLASS_TO_ID["road"]).any()
    assert (lab[6:, :] == 0).all()      # dropped class stays background
