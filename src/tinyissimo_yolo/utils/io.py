import os

from tinyissimo_yolo._constants import IMAGE_EXTENSIONS, TILE_SEPARATOR


def load_tiled_test_images(dirpath):
    images = {}
    for filename in os.listdir(dirpath):
        if not filename.endswith(IMAGE_EXTENSIONS):
            continue
        full_image_name = filename[: -(len(filename.split(TILE_SEPARATOR)[-1]) + 1)] + '.' + filename.split('.')[-1]
        images.setdefault(full_image_name, []).append(os.path.join(dirpath, filename))
    for image in images:
        images[image] = sorted(images[image])
    return images


def load_test_images(dirpath):
    images = {}
    for filename in os.listdir(dirpath):
        if filename.endswith(IMAGE_EXTENSIONS):
            images[filename] = [os.path.join(dirpath, filename)]
    for image in images:
        images[image] = sorted(images[image])
    return images
