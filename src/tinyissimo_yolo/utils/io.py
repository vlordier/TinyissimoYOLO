import os

from tinyissimo_yolo._constants import IMAGE_EXTENSIONS, TILE_SEPARATOR


def _base_image_name(filename: str) -> str:
    """Extract the original image name from a tiled filename.

    ``scene1_0.png`` → ``scene1.png``, ``image.jpg`` → ``image.jpg``.
    """
    if TILE_SEPARATOR not in filename:
        return filename
    name_no_ext, ext = filename.rsplit('.', 1)
    base = name_no_ext.rsplit(TILE_SEPARATOR, 1)[0]
    return f'{base}.{ext}'


def load_tiled_test_images(dirpath: str) -> dict[str, list[str]]:
    """Load tiled images grouped by original image name.

    Returns ``{original_name: [list of tiled paths]}``.
    """
    images: dict[str, list[str]] = {}
    for filename in os.listdir(dirpath):
        if not filename.endswith(IMAGE_EXTENSIONS):
            continue
        key = _base_image_name(filename)
        images.setdefault(key, []).append(os.path.join(dirpath, filename))
    return {k: sorted(v) for k, v in images.items()}


def load_test_images(dirpath: str) -> dict[str, list[str]]:
    """Load non-tiled test images.

    Returns ``{filename: [single path]}``.
    """
    images: dict[str, list[str]] = {}
    for filename in os.listdir(dirpath):
        if filename.endswith(IMAGE_EXTENSIONS):
            images[filename] = [os.path.join(dirpath, filename)]
    return {k: sorted(v) for k, v in images.items()}
