from tinyissimo_yolo.utils.dataset import load_gt_bbox


def test_load_gt_bbox(tmp_path):
    label_file = tmp_path / 'label.txt'
    # format: x1 y1 x2 y2 (class id treated as x1 in original impl)
    label_file.write_text('10 20 110 120 200\n0 50 60 80 90\n')
    annots = load_gt_bbox(str(label_file))
    assert len(annots) == 2
    for a in annots:
        assert a['label'] == 'car'
        assert set(a['coordinates']) == {'x', 'y', 'width', 'height'}
    # first annotation: x1=10, y1=20, x2=110, y2=120
    w1, h1 = annots[0]['coordinates']['width'], annots[0]['coordinates']['height']
    assert w1 == 100
    assert h1 == 100


def test_load_gt_bbox_empty(tmp_path):
    label_file = tmp_path / 'empty.txt'
    label_file.write_text('')
    assert load_gt_bbox(str(label_file)) == []


def test_load_gt_bbox_invalid(tmp_path):
    label_file = tmp_path / 'bad.txt'
    label_file.write_text('not a valid label\n')
    assert load_gt_bbox(str(label_file)) == []
