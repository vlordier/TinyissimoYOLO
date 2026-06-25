import cv2
import torch
import torchvision.ops as ops

from tinyissimo_yolo._constants import (
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    RECT_BOLD,
)


def compute_metrics(
    gt, pred_boxes, pred_conf, og_image, conf_thresh=DEFAULT_CONF_THRESH, iou_thresh=DEFAULT_IOU_THRESH, plot=False
):
    pred = torch.tensor(pred_boxes)
    pred_conf = torch.tensor(pred_conf)
    pred = pred[pred_conf > conf_thresh]

    gt_boxes = torch.zeros((len(gt), 4))
    img_w, img_h = og_image.shape[1], og_image.shape[0]
    for i, instance in enumerate(gt):
        xc, yc, w, h = instance[1:5]
        xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
        x1 = xc - w * 0.5
        y1 = yc - h * 0.5
        x2 = xc + w * 0.5
        y2 = yc + h * 0.5
        gt_boxes[i] = torch.tensor([x1, y1, x2, y2])

    iou = ops.box_iou(gt_boxes, pred)
    gt_has_match = iou.amax(dim=1) > iou_thresh
    pred_has_match = iou.amax(dim=0) > iou_thresh
    tp = pred_has_match.sum()
    fp = (~pred_has_match).sum()
    fn = (~gt_has_match).sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    num_gt = len(gt)
    num_pred = len(pred)
    count_mae = abs(num_gt - num_pred)

    if plot:
        filtered_image = og_image.copy()
        for i, instance in enumerate(gt_boxes):
            x1, y1, x2, y2 = instance
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if gt_has_match[i]:
                continue
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), COLOR_RED, RECT_BOLD)
        for i, box in enumerate(pred):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = COLOR_GREEN if pred_has_match[i] else COLOR_BLUE
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), color, RECT_BOLD)

        cv2.imshow('filtered image', filtered_image)
        cv2.waitKey(0)

    return count_mae, precision, recall, f1
