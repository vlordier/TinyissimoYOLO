import os
import re
import urllib.request

import cv2


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
        x = x1 + 0.5 * width
        y = y1 + 0.5 * height
        instance = {'label': 'car', 'coordinates': {'x': x, 'y': y, 'width': int(width), 'height': int(height)}}
        annots.append(instance)
    return annots


def plot_bboxes(image, instances):
    image_plot = image.copy()
    for instance in instances:
        width = instance['coordinates']['width']
        height = instance['coordinates']['height']
        x = int(instance['coordinates']['x'] - 0.5 * width)
        y = int(instance['coordinates']['y'] - 0.5 * height)
        start_point = (x, y)
        end_point = (x + width, y + height)
        color = (255, 0, 0)
        thickness = 2
        image_plot = cv2.rectangle(image_plot, start_point, end_point, color, thickness)

    cv2.imshow('annotated image', image_plot)
    cv2.waitKey(0)


def convert_carpk_to_create_ml(label_dir, images_dir, debug_plot=False):
    label_list = []
    for image_filename in os.listdir(images_dir):
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


def _download_split(url):
    """Download a split file from GitHub and return list of image names (without extension)."""
    lines = []
    for line in urllib.request.urlopen(url):
        lines.append(line.decode('utf-8').split('.')[0].strip())
    return lines


def convert_create_ml_to_yolo(labels, image_dir, parent_dir):
    train_split = _download_split('https://github.com/mojulian/ultralytics/releases/download/0.1/train_images.txt')
    val_split = _download_split('https://github.com/mojulian/ultralytics/releases/download/0.1/val_images.txt')
    test_split = _download_split('https://github.com/mojulian/ultralytics/releases/download/0.1/test.txt')

    for folder in ('CARPK_train', 'CARPK_val', 'CARPK_test'):
        os.makedirs(os.path.join(parent_dir, folder, 'images'), exist_ok=True)
        os.makedirs(os.path.join(parent_dir, folder, 'annotations'), exist_ok=True)

    for image in labels:
        image_name = image['image']
        image_name_wo_extension = image_name.split('.')[0]
        image_path = os.path.join(image_dir, image['image'])
        img = cv2.imread(image_path)
        img_res = img.shape[:2]

        yolo_annotations = ''

        for annot in image['annotations']:
            if annot['label'] != 'car':
                print(f'Found an annotation with label {annot["label"]}. Skipping...')
                continue

            obj_class = 0
            x = annot['coordinates']['x']
            y = annot['coordinates']['y']
            width = annot['coordinates']['width']
            height = annot['coordinates']['height']

            x_center = round(x / img_res[1], 6)
            y_center = round(y / img_res[0], 6)
            w = round(width / img_res[1], 6)
            h = round(height / img_res[0], 6)

            yolo_annotations += f'{obj_class} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n'

        if image_name_wo_extension in train_split:
            folder = 'CARPK_train'
        elif image_name_wo_extension in val_split:
            folder = 'CARPK_val'
        elif image_name_wo_extension in test_split:
            folder = 'CARPK_test'
        else:
            continue

        dst_image_path = os.path.join(parent_dir, folder, 'images', image['image'])
        cv2.imwrite(dst_image_path, img)

        annot_file_path = os.path.join(parent_dir, folder, 'annotations', image_name_wo_extension + '.txt')
        with open(annot_file_path, 'w') as f:
            f.writelines(yolo_annotations)

        print(f'Created annotation file for {image["image"]}')
