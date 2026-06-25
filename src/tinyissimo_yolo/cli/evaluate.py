import argparse
import os

import numpy as np
import yaml

import tinyissimo_yolo._vendored  # noqa: F401
from tinyissimo_yolo._constants import (
    DATASET_YAML,
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    IOU_SWEEP_END,
    IOU_SWEEP_START,
    IOU_SWEEP_STEPS,
    TILING_CONFIG,
)
from tinyissimo_yolo._logging import get_logger
from tinyissimo_yolo.utils.io import load_test_images, load_tiled_test_images
from tinyissimo_yolo.utils.metrics import compute_metrics

log = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate YOLO model on tiled or full images', add_help=False)
    parser.add_argument('--use-tiling', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--perform-iou-sweep', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--plot', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--image-set', default='test')
    parser.add_argument('--dataset-yaml-path', default=DATASET_YAML)
    parser.add_argument('--model-path', default='path/to/your/model.pt')
    parser.add_argument('--tiling-config', default=TILING_CONFIG)
    parser.add_argument('--conf-thresh', type=float, default=DEFAULT_CONF_THRESH)
    parser.add_argument('--iou-thresh', type=float, default=DEFAULT_IOU_THRESH)
    return parser


def _load_dataset_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_labels(labels_dir: str, image_name: str) -> list[list[float]]:
    path = os.path.join(labels_dir, image_name.replace('png', 'txt'))
    with open(path) as f:
        lines = f.read().splitlines()
    return [[float(v) for v in line.split()] for line in lines]


def _collect_predictions(result) -> tuple[list, list]:
    boxes, confs = [], []
    for pred in result:
        for box, conf in zip(pred.boxes.xyxy, pred.boxes.conf):
            x1, y1, x2, y2 = box
            boxes.append([x1, y1, x2, y2])
            confs.append(conf)
    return boxes, confs


def main() -> None:
    parser = get_parser()
    parser.add_help = True
    args = parser.parse_args()

    import cv2
    from tqdm import tqdm

    from ultralytics import YOLO
    from ultralytics.utils.offline_tiling import Tiler

    dataset_yaml = _load_dataset_yaml(args.dataset_yaml_path)
    data_dir: str = dataset_yaml['path'] + '/'

    if args.use_tiling:
        tiler = Tiler(args.tiling_config)
        tiler.get_split_dataset()
        test_images = load_tiled_test_images(data_dir + dataset_yaml[args.image_set])
        with open(data_dir + dataset_yaml[args.image_set].replace('images', 'tiles_dict') + '.yaml') as f:
            tiles_dict = yaml.safe_load(f)
    else:
        test_images = load_test_images(data_dir + dataset_yaml['original_images'][args.image_set])

    original_image_dir: str = data_dir + dataset_yaml['original_images'][args.image_set]
    original_labels_dir: str = original_image_dir.replace('images', 'annotations')

    model = YOLO(args.model_path)

    full_mae: list[float] = []
    full_prec: list[float] = []
    full_rec: list[float] = []
    full_f1: list[float] = []

    for image in tqdm(test_images, position=0, leave=True):
        og_image = cv2.imread(os.path.join(original_image_dir, image))
        instances_float = _load_labels(original_labels_dir, image)

        result = model(test_images[image], stream=True, verbose=False)

        if args.use_tiling:
            _, boxes, confs = tiler.stitch_tiled_predictions(
                result,  # type: ignore[arg-type]
                tiles_dict,
                image,
            )
        else:
            boxes, confs = _collect_predictions(result)

        iou_thresh_vals = (
            np.linspace(IOU_SWEEP_START, IOU_SWEEP_END, IOU_SWEEP_STEPS)
            if args.perform_iou_sweep
            else [args.iou_thresh]
        )
        for thresh in iou_thresh_vals:
            mae, pr, re, f1 = compute_metrics(
                instances_float,
                boxes,
                confs,
                og_image,  # type: ignore[arg-type]
                conf_thresh=args.conf_thresh,
                iou_thresh=thresh,
                plot=args.plot,
            )
            full_mae.append(mae)
            full_prec.append(pr)
            full_rec.append(re)
            full_f1.append(f1)

    log.info(
        'Average Count Mae: %s, Average Precision: %s, Average Recall: %s, Average F1: %s',
        np.nanmean(full_mae),
        np.nanmean(full_prec),
        np.mean(full_rec),
        np.nanmean(full_f1),
    )


if __name__ == '__main__':
    main()
