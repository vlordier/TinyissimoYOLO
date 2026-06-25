import numpy as np

from tinyissimo_yolo.utils.metrics import compute_metrics


def _dummy_image(height=100, width=200):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _gt_box(xc_norm, yc_norm, w_norm, h_norm, img_w=200, img_h=100):
    """Convert normalized GT coords to pixel x1,y1,x2,y2."""
    xc, yc = xc_norm * img_w, yc_norm * img_h
    w, h = w_norm * img_w, h_norm * img_h
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def test_compute_metrics_perfect_match():
    img = _dummy_image()
    gt = [[0, 0.5, 0.5, 0.4, 0.4]]  # class, xc, yc, w, h (normalized)
    # GT box in pixels: (60, 30) - (140, 70) on 200x100 image
    pred_boxes = [_gt_box(0.5, 0.5, 0.4, 0.4)]
    pred_conf = [0.9]
    mae, prec, rec, f1 = compute_metrics(gt, pred_boxes, pred_conf, img, conf_thresh=0.5, iou_thresh=0.3)
    assert mae == 0
    assert prec.item() == 1.0
    assert rec.item() == 1.0
    assert f1.item() == 1.0


def test_compute_metrics_no_match():
    img = _dummy_image()
    gt = [[0, 0.5, 0.5, 0.4, 0.4]]
    pred_boxes = [[0, 0, 5, 5]]
    pred_conf = [0.9]
    mae, prec, rec, _ = compute_metrics(gt, pred_boxes, pred_conf, img, conf_thresh=0.5, iou_thresh=0.5)
    assert mae == 0  # 1 GT, 1 pred → count matches
    assert prec.item() == 0.0
    assert rec.item() == 0.0


def test_compute_metrics_count_mismatch():
    img = _dummy_image()
    gt = [[0, 0.5, 0.5, 0.4, 0.4], [0, 0.1, 0.1, 0.2, 0.2]]
    pred_boxes = [[0, 0, 5, 5]]
    pred_conf = [0.9]
    mae, _, _, _ = compute_metrics(gt, pred_boxes, pred_conf, img, conf_thresh=0.5, iou_thresh=0.5)
    assert mae == 1  # 2 GT, 1 pred


def test_compute_metrics_filters_low_conf():
    img = _dummy_image()
    gt = [[0, 0.5, 0.5, 0.4, 0.4]]
    # one good pred + one low-confidence pred that should be filtered
    pred_boxes = [
        _gt_box(0.5, 0.5, 0.4, 0.4),
        [0, 0, 5, 5],
    ]
    pred_conf = [0.9, 0.1]
    mae, prec, rec, _ = compute_metrics(gt, pred_boxes, pred_conf, img, conf_thresh=0.5, iou_thresh=0.3)
    assert mae == 0
    assert prec.item() == 1.0
    assert rec.item() == 1.0
