import os
import re
import urllib.request

import cv2

from tinyissimo_yolo._constants import (
    CAR_CLASS_ID,
    CAR_LABEL,
    CARPK_FOLDERS,
    COLOR_RED,
    HALF,
    IMAGE_EXTENSIONS,
    RECT_NORMAL,
    SPLIT_URLS,
    YOLO_ROUND_DECIMALS,
)


def load_gt_bbox(filepath):
    with open(filepath) as f:
        data = f.read()
    objs = re.findall(r'\d+ \d+ \d+ \d+ \d+', data)
    annots = []
    for obj in objs:
        info = re.findall(r'\d+', obj)
        x1 = float(info[0])
        y1 = float(info[1])
        x2 = float(info[2])
        y2 = float(info[3])
        width = x2 - x1
        height = y2 - y1
        x = x1 + HALF * width
        y = y1 + HALF * height
        instance = {
            'label': CAR_LABEL,
            'coordinates': {'x': x, 'y': y, 'width': int(width), 'height': int(height)},
        }
        annots.append(instance)
    return annots


def plot_bboxes(image, instances):
    image_plot = image.copy()
    for instance in instances:
        width = instance['coordinates']['width']
        height = instance['coordinates']['height']
        x = int(instance['coordinates']['x'] - HALF * width)
        y = int(instance['coordinates']['y'] - HALF * height)
        start_point = (x, y)
        end_point = (x + width, y + height)
        image_plot = cv2.rectangle(image_plot, start_point, end_point, COLOR_RED, RECT_NORMAL)

    cv2.imshow('annotated image', image_plot)
    cv2.waitKey(0)


def convert_carpk_to_create_ml(label_dir, images_dir, debug_plot=False):
    label_list = []
    for image_filename in os.listdir(images_dir):
        if not image_filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        base_filename = image_filename.strip().split('.')[0]
        annot_filename = base_filename + '.txt'
        annotations = load_gt_bbox(os.path.join(label_dir, annot_filename))
        image_dict = {
            'image': image_filename,
            'annotations': annotations,
            'normalized_avg_bbox_area': -1,
            'overlapping_bboxes_exist': True,
            'top_down_view': True,
        }
        label_list.append(image_dict)

        if debug_plot and image_filename == '20160331_NTU_00066.png':
            img = cv2.imread(os.path.join(images_dir, image_filename))
            plot_bboxes(img, image_dict['annotations'])

    return label_list


def _download_split(key):
    """Download a split file from GitHub and return list of image names (without extension)."""
    url = SPLIT_URLS[key]
    return [line.decode('utf-8').split('.')[0].strip() for line in urllib.request.urlopen(url)]


def _img_resolution(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {image_path}')
    return img.shape[:2]


def convert_create_ml_to_yolo(labels, image_dir, parent_dir):
    train_split = _download_split('train')
    val_split = _download_split('val')
    test_split = _download_split('test')

    for folder in CARPK_FOLDERS:
        os.makedirs(os.path.join(parent_dir, folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, folder, 'annotations'), exist_ok=True)

    for image in labels:
        image_name = image['image']
        image_name_wo_extension = image_name.split('.')[0]
        image_path = os.path.join(image_dir, image['image'])
        img_h, img_w = _img_resolution(image_path)

        yolo_annotations = ''

        for annot in image['annotations']:
            if annot['label'] != CAR_LABEL:
                print(f'Found an annotation with label {annot["label"]}. Skipping...')
                continue

            x = annot['coordinates']['x']
            y = annot['coordinates']['y']
            width = annot['coordinates']['width']
            height = annot['coordinates']['height']

            x_center = round(x / img_w, YOLO_ROUND_DECIMALS)
            y_center = round(y / img_h, YOLO_ROUND_DECIMALS)
            w = round(width / img_w, YOLO_ROUND_DECIMALS)
            h = round(height / img_h, YOLO_ROUND_DECIMALS)

            yolo_annotations += f'{CAR_CLASS_ID} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n'

        if image_name_wo_extension in train_split:
            folder = CARPK_FOLDERS[0]
        elif image_name_wo_extension in val_split:
            folder = CARPK_FOLDERS[1]
        elif image_name_wo_extension in test_split:
            folder = CARPK_FOLDERS[2]
        else:
            continue

        dst_image_path = os.path.join(parent_dir, folder, 'images', image['image'])
        cv2.imwrite(dst_image_path, cv2.imread(image_path))

        annot_file_path = os.path.join(parent_dir, folder, 'annotations', image_name_wo_extension + '.txt')
        with open(annot_file_path, 'w') as f:
            f.writelines(yolo_annotations)

        print(f'Created annotation file for {image["image"]}')
