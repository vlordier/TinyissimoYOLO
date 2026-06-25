import cv2
import numpy as np
import torch
import torchvision.ops as ops

from tinyissimo_yolo._constants import (
    CIRCLE_FILL,
    CIRCLE_RADIUS,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    HALF,
    RECT_BOLD,
    RECT_NORMAL,
    RECT_THICK,
    RECT_THIN,
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
        x1 = xc - w * HALF
        y1 = yc - h * HALF
        x2 = xc + w * HALF
        y2 = yc + h * HALF
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


def plot_results(stitched_preds, filtered_boxes, filtered_conf, og_image, og_labels, conf_thresh=DEFAULT_CONF_THRESH):
    filtered_image = og_image.copy()
    for tile_idx in stitched_preds:
        tile = stitched_preds[tile_idx]['tile']
        instances = stitched_preds[tile_idx]['predictions']
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(og_image, (tile['x_min'], tile['y_min']), (tile['x_max'], tile['y_max']), color, RECT_NORMAL)

        for instance in instances:
            x1, y1, x2, y2 = instance['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(og_image, (x1, y1), (x2, y2), color, RECT_NORMAL)
        cv2.imshow('stitched image', og_image)
        cv2.waitKey(0)

    for instance in og_labels:
        xc, yc, w, h = instance[1:5]
        xc = int(xc * og_image.shape[1])
        yc = int(yc * og_image.shape[0])
        w = int(w * og_image.shape[1])
        h = int(h * og_image.shape[0])
        x1 = xc - w // 2
        y1 = yc - h // 2
        x2 = xc + w // 2
        y2 = yc + h // 2
        cv2.rectangle(filtered_image, (x1, y1), (x2, y2), COLOR_BLUE, RECT_THIN)
        cv2.circle(filtered_image, (xc, yc), CIRCLE_RADIUS, COLOR_BLUE, CIRCLE_FILL)
    for box, conf in zip(filtered_boxes, filtered_conf):
        if conf > conf_thresh:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), COLOR_GREEN, RECT_THICK)

    cv2.imshow('filtered image', filtered_image)
    cv2.waitKey(0)
