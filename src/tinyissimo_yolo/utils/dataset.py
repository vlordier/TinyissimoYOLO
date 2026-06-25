import os
import re
import sys
import urllib.request
from typing import Any

import cv2

from tinyissimo_yolo._constants import (
    CAR_CLASS_ID,
    CAR_LABEL,
    CARPK_FOLDERS,
    COLOR_RED,
    IMAGE_EXTENSIONS,
    RECT_NORMAL,
    SPLIT_URLS,
    YOLO_ROUND_DECIMALS,
)
from tinyissimo_yolo._logging import get_logger

log = get_logger(__name__)

_ANNOT_RE: re.Pattern = re.compile(r'\d+ \d+ \d+ \d+ \d+')
_HAS_DISPLAY: bool = os.environ.get('DISPLAY') is not None or sys.platform != 'linux'


def load_gt_bbox(filepath: str) -> list[dict[str, Any]]:
    """Load CARPK annotation file.

    Format: ``<class> <x1> <y1> <x2> <y2>`` per line.
    """
    with open(filepath) as f:
        data = f.read()
    objs = _ANNOT_RE.findall(data)
    annots: list[dict[str, Any]] = []
    for obj in objs:
        info = re.findall(r'\d+', obj)
        x1 = float(info[1])
        y1 = float(info[2])
        x2 = float(info[3])
        y2 = float(info[4])
        width = x2 - x1
        height = y2 - y1
        annots.append(
            {
                'label': CAR_LABEL,
                'coordinates': {
                    'x': x1 + 0.5 * width,
                    'y': y1 + 0.5 * height,
                    'width': int(width),
                    'height': int(height),
                },
            }
        )
    return annots


def plot_bboxes(image: cv2.Mat, instances: list[dict[str, Any]]) -> None:
    """Overlay bounding boxes on *image* and show it (blocks, headless-safe)."""
    overlay = image.copy()
    for inst in instances:
        c = inst['coordinates']
        x = int(c['x'] - 0.5 * c['width'])
        y = int(c['y'] - 0.5 * c['height'])
        cv2.rectangle(overlay, (x, y), (x + c['width'], y + c['height']), COLOR_RED, RECT_NORMAL)
    if _HAS_DISPLAY:
        cv2.imshow('annotated image', overlay)
        cv2.waitKey(0)


def convert_carpk_to_create_ml(label_dir: str, images_dir: str, debug_plot: bool = False) -> list[dict[str, Any]]:
    """Convert CARPK annotations to CreateML-style label list."""
    label_list: list[dict[str, Any]] = []
    for image_filename in os.listdir(images_dir):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        base = image_filename.strip().split('.')[0]
        annotations = load_gt_bbox(os.path.join(label_dir, base + '.txt'))
        label_list.append({'image': image_filename, 'annotations': annotations})
        if debug_plot:
            img = cv2.imread(os.path.join(images_dir, image_filename))
            plot_bboxes(img, annotations)
    return label_list


def _download_split(key: str) -> list[str]:
    """Download a split file from GitHub; return image names (without extension)."""
    return [line.decode('utf-8').split('.')[0].strip() for line in urllib.request.urlopen(SPLIT_URLS[key])]


def _img_data(image_path: str) -> tuple[int, int, cv2.Mat]:
    """Return ``(height, width, image_array)`` for *image_path*."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    h, w = img.shape[:2]
    return h, w, img


def convert_create_ml_to_yolo(labels: list[dict[str, Any]], image_dir: str, parent_dir: str) -> None:
    """Convert CreateML label list to YOLO-format dataset on disk."""
    train_split = _download_split('train')
    val_split = _download_split('val')
    test_split = _download_split('test')

    for folder in CARPK_FOLDERS:
        os.makedirs(os.path.join(parent_dir, folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, folder, 'annotations'), exist_ok=True)

    for image in labels:
        image_name: str = image['image']
        stem: str = image_name.split('.')[0]
        image_path: str = os.path.join(image_dir, image_name)
        img_h, img_w, img = _img_data(image_path)

        yolo_lines: list[str] = []
        for annot in image['annotations']:
            if annot['label'] != CAR_LABEL:
                log.warning('Skipping label %s', annot['label'])
                continue
            c = annot['coordinates']
            yolo_lines.append(
                f'{CAR_CLASS_ID} {round(c["x"] / img_w, YOLO_ROUND_DECIMALS):.6f} '
                f'{round(c["y"] / img_h, YOLO_ROUND_DECIMALS):.6f} '
                f'{round(c["width"] / img_w, YOLO_ROUND_DECIMALS):.6f} '
                f'{round(c["height"] / img_h, YOLO_ROUND_DECIMALS):.6f}\n'
            )

        if stem in train_split:
            folder = CARPK_FOLDERS[0]
        elif stem in val_split:
            folder = CARPK_FOLDERS[1]
        elif stem in test_split:
            folder = CARPK_FOLDERS[2]
        else:
            continue

        cv2.imwrite(os.path.join(parent_dir, folder, 'images', image_name), img)

        annot_path: str = os.path.join(parent_dir, folder, 'annotations', stem + '.txt')
        with open(annot_path, 'w') as f:
            f.writelines(yolo_lines)

        log.info('Created annotation file for %s', image_name)
