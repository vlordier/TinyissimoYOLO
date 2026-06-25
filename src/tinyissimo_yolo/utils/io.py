import os


def load_tiled_test_images(dirpath):
    images = {}
    for filename in os.listdir(dirpath):
        if not (filename.endswith('.jpg') or filename.endswith('.png')):
            continue
        full_image_name = filename[: -(len(filename.split('_')[-1]) + 1)] + '.' + filename.split('.')[-1]
        images.setdefault(full_image_name, []).append(os.path.join(dirpath, filename))
    for image in images:
        images[image] = sorted(images[image])
    return images


def load_test_images(dirpath):
    images = {}
    for filename in os.listdir(dirpath):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            images[filename] = [os.path.join(dirpath, filename)]
    for image in images:
        images[image] = sorted(images[image])
    return images
