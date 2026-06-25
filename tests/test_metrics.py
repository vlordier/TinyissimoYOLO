from tinyissimo_yolo.utils.metrics import compute_metrics

from .conftest import gt_box


def test_compute_metrics_perfect_match(dummy_image):
    gt = [[0, 0.5, 0.5, 0.4, 0.4]]
    pred_boxes = [gt_box(0.5, 0.5, 0.4, 0.4)]
    pred_conf = [0.9]
    mae, prec, rec, f1 = compute_metrics(gt, pred_boxes, pred_conf, dummy_image, conf_thresh=0.5, iou_thresh=0.3)
    assert mae == 0
    assert prec.item() == 1.0
    assert rec.item() == 1.0
    assert f1.item() == 1.0


def test_compute_metrics_no_match(dummy_image):
    gt = [[0, 0.5, 0.5, 0.4, 0.4]]
    pred_boxes = [[0, 0, 5, 5]]
    pred_conf = [0.9]
    mae, prec, rec, _ = compute_metrics(gt, pred_boxes, pred_conf, dummy_image, conf_thresh=0.5, iou_thresh=0.5)
    assert mae == 0
    assert prec.item() == 0.0
    assert rec.item() == 0.0


def test_compute_metrics_count_mismatch(dummy_image):
    gt = [[0, 0.5, 0.5, 0.4, 0.4], [0, 0.1, 0.1, 0.2, 0.2]]
    pred_boxes = [[0, 0, 5, 5]]
    pred_conf = [0.9]
    mae, _, _, _ = compute_metrics(gt, pred_boxes, pred_conf, dummy_image, conf_thresh=0.5, iou_thresh=0.5)
    assert mae == 1


def test_compute_metrics_filters_low_conf(dummy_image):
    gt = [[0, 0.5, 0.5, 0.4, 0.4]]
    pred_boxes = [gt_box(0.5, 0.5, 0.4, 0.4), [0, 0, 5, 5]]
    pred_conf = [0.9, 0.1]
    mae, prec, rec, _ = compute_metrics(gt, pred_boxes, pred_conf, dummy_image, conf_thresh=0.5, iou_thresh=0.3)
    assert mae == 0
    assert prec.item() == 1.0
    assert rec.item() == 1.0
