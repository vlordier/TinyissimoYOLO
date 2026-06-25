import os

from tinyissimo_yolo._constants import IMAGE_EXTENSIONS, TILE_SEPARATOR


def _base_image_name(filename):
    """Extract the original image name from a tiled filename.

    For 'scene1_0.png' returns 'scene1.png'.
    For non-tiled 'image.png' returns 'image.png'.
    """
    if TILE_SEPARATOR not in filename:
        return filename
    name_no_ext, ext = filename.rsplit('.', 1)
    base = name_no_ext.rsplit(TILE_SEPARATOR, 1)[0]
    return f'{base}.{ext}'


def load_tiled_test_images(dirpath):
    images = {}
    for filename in os.listdir(dirpath):
        if not filename.endswith(IMAGE_EXTENSIONS):
            continue
        full_image_name = _base_image_name(filename)
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
