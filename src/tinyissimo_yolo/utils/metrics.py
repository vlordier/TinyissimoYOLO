import cv2
import numpy as np
import torch
import torchvision.ops as ops


def compute_metrics(gt, pred_boxes, pred_conf, og_image, conf_thresh=0.6, iou_thresh=0.5, plot=False):
    pred = torch.tensor(pred_boxes)
    pred_conf = torch.tensor(pred_conf)
    pred = pred[pred_conf > conf_thresh]

    gt_boxes = torch.zeros((len(gt), 4))
    img_w, img_h = og_image.shape[1], og_image.shape[0]
    for i, instance in enumerate(gt):
        xc, yc, w, h = instance[1:5]
        xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
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
            color = (255, 0, 0)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), color, 3)
        for i, box in enumerate(pred):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if pred_has_match[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), color, 3)

        cv2.imshow('filtered image', filtered_image)
        cv2.waitKey(0)

    return count_mae, precision, recall, f1


def plot_results(stitched_preds, filtered_boxes, filtered_conf, og_image, og_labels, conf_thresh=0.6):
    filtered_image = og_image.copy()
    for tile_idx in stitched_preds:
        tile = stitched_preds[tile_idx]['tile']
        instances = stitched_preds[tile_idx]['predictions']
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(og_image, (tile['x_min'], tile['y_min']), (tile['x_max'], tile['y_max']), color, 2)

        for instance in instances:
            x1, y1, x2, y2 = instance['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(og_image, (x1, y1), (x2, y2), color, 2)
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
        cv2.rectangle(filtered_image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.circle(filtered_image, (xc, yc), 5, (0, 0, 255), -1)
    for box, conf in zip(filtered_boxes, filtered_conf):
        if conf > conf_thresh:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(filtered_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    cv2.imshow('filtered image', filtered_image)
    cv2.waitKey(0)
