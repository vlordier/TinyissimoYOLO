from tinyissimo_yolo.utils.io import load_test_images, load_tiled_test_images


def _create_image(dirpath, name):
    (dirpath / name).touch()


def test_load_test_images(tmp_path):
    _create_image(tmp_path, 'img001.png')
    _create_image(tmp_path, 'img002.jpg')
    _create_image(tmp_path, 'notes.txt')
    result = load_test_images(str(tmp_path))
    assert set(result.keys()) == {'img001.png', 'img002.jpg'}
    assert result['img001.png'] == [str(tmp_path / 'img001.png')]


def test_load_tiled_test_images(tmp_path):
    _create_image(tmp_path, 'scene1_0.png')
    _create_image(tmp_path, 'scene1_1.png')
    _create_image(tmp_path, 'scene2_0.png')
    result = load_tiled_test_images(str(tmp_path))
    assert set(result.keys()) == {'scene1.png', 'scene2.png'}
    assert len(result['scene1.png']) == 2
    assert len(result['scene2.png']) == 1
    # tiles are sorted
    assert result['scene1.png'] == sorted(result['scene1.png'])


def test_load_test_images_empty_dir(tmp_path):
    assert load_test_images(str(tmp_path)) == {}
