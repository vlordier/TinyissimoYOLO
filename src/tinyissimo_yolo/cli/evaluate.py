import argparse
import os

import numpy as np
import yaml

import tinyissimo_yolo._vendored  # noqa: F401  ensure vendored ultralytics is on sys.path
from tinyissimo_yolo._constants import (
    DATASET_YAML,
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    IOU_SWEEP_END,
    IOU_SWEEP_START,
    IOU_SWEEP_STEPS,
    TILING_CONFIG,
)
from tinyissimo_yolo.utils.io import load_test_images, load_tiled_test_images
from tinyissimo_yolo.utils.metrics import compute_metrics


def _load_labels(labels_dir, image_name):
    """Load ground-truth labels for a single image."""
    path = os.path.join(labels_dir, image_name.replace('png', 'txt'))
    with open(path) as f:
        lines = f.read().splitlines()
    instances = [line.split(' ') for line in lines]
    return [[float(v) for v in inst] for inst in instances]


def _collect_predictions(result):
    boxes, confs = [], []
    for pred in result:
        for box, conf in zip(pred.boxes.xyxy, pred.boxes.conf):
            x1, y1, x2, y2 = box
            boxes.append([x1, y1, x2, y2])
            confs.append(conf)
    return boxes, confs


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on tiled or full images')
    parser.add_argument('--use-tiling', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--perform-iou-sweep', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--plot', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--image-set', default='test')
    parser.add_argument('--dataset-yaml-path', default=DATASET_YAML)
    parser.add_argument('--model-path', default='path/to/your/model.pt')
    parser.add_argument('--tiling-config', default=TILING_CONFIG)
    parser.add_argument('--conf-thresh', type=float, default=DEFAULT_CONF_THRESH)
    parser.add_argument('--iou-thresh', type=float, default=DEFAULT_IOU_THRESH)
    args = parser.parse_args()

    import cv2
    from tqdm import tqdm

    from ultralytics import YOLO
    from ultralytics.utils.offline_tiling import Tiler

    if args.use_tiling:
        tiler = Tiler(args.tiling_config)
        tiler.get_split_dataset()
        with open(args.dataset_yaml_path) as f:
            dataset_yaml = yaml.safe_load(f)
        data_dir = dataset_yaml['path'] + '/'
        test_images = load_tiled_test_images(data_dir + dataset_yaml[args.image_set])
        with open(data_dir + dataset_yaml[args.image_set].replace('images', 'tiles_dict') + '.yaml') as f:
            tiles_dict = yaml.safe_load(f)
    else:
        with open(args.dataset_yaml_path) as f:
            dataset_yaml = yaml.safe_load(f)
        data_dir = dataset_yaml['path'] + '/'
        test_images = load_test_images(data_dir + dataset_yaml['original_images'][args.image_set])

    original_image_dir = data_dir + dataset_yaml['original_images'][args.image_set]
    original_labels_dir = data_dir + dataset_yaml['original_images'][args.image_set].replace('images', 'annotations')

    model = YOLO(args.model_path)

    full_count_mae = []
    full_precision = []
    full_recall = []
    full_f1 = []

    for image in tqdm(test_images, position=0, leave=True):
        og_image = cv2.imread(os.path.join(original_image_dir, image))
        instances_float = _load_labels(original_labels_dir, image)

        result = model(test_images[image], stream=True, verbose=False)

        if args.use_tiling:
            _stitched_preds, filtered_boxes, filtered_conf = tiler.stitch_tiled_predictions(result, tiles_dict, image)
        else:
            filtered_boxes, filtered_conf = _collect_predictions(result)

        if args.perform_iou_sweep:
            iou_thresh_vals = np.linspace(IOU_SWEEP_START, IOU_SWEEP_END, IOU_SWEEP_STEPS)
            for iou_thresh in iou_thresh_vals:
                count_mae, pr, re, f1 = compute_metrics(
                    instances_float,
                    filtered_boxes,
                    filtered_conf,
                    og_image,
                    conf_thresh=args.conf_thresh,
                    iou_thresh=iou_thresh,
                    plot=args.plot,
                )
                full_count_mae.append(count_mae)
                full_precision.append(pr)
                full_recall.append(re)
                full_f1.append(f1)
        else:
            count_mae, pr, re, f1 = compute_metrics(
                instances_float,
                filtered_boxes,
                filtered_conf,
                og_image,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh,
                plot=args.plot,
            )
            full_count_mae.append(count_mae)
            full_precision.append(pr)
            full_recall.append(re)
            full_f1.append(f1)

    print(
        f'Average Count Mae: {np.nanmean(full_count_mae)},'
        f' Average Precision: {np.nanmean(full_precision)},'
        f' Average Recall: {np.mean(full_recall)},'
        f' Average F1: {np.nanmean(full_f1)}'
    )


if __name__ == '__main__':
    main()
