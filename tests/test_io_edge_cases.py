from tinyissimo_yolo.utils.io import _base_image_name, load_tiled_test_images


def _touch(dirpath, name):
    (dirpath / name).touch()


def test_base_image_name_tiled():
    assert _base_image_name('scene1_0.png') == 'scene1.png'
    assert _base_image_name('frame_001_2.jpg') == 'frame_001.jpg'


def test_base_image_name_untiled():
    assert _base_image_name('image.png') == 'image.png'
    assert _base_image_name('single.jpg') == 'single.jpg'


def test_load_tiled_with_untiled_file(tmp_path):
    """A non-tiled .png in a tiled directory should appear as its own image."""
    _touch(tmp_path, 'single.png')
    result = load_tiled_test_images(str(tmp_path))
    assert 'single.png' in result
    assert len(result['single.png']) == 1
