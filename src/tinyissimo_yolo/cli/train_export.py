import argparse

import tinyissimo_yolo._vendored  # noqa: F401
from tinyissimo_yolo._constants import (
    DEFAULT_BATCH_LARGE,
    DEFAULT_EPOCHS_LONG,
    DEFAULT_EXP_NAME,
    DEFAULT_IMGSZ,
    DEFAULT_PROJECT,
    DEFAULT_SGD,
)
from tinyissimo_yolo._logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train and export a TinyissimoYOLO model')
    parser.add_argument('--version', default='v8', choices=['v1', 'v8'])
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--exp-id', default=DEFAULT_EXP_NAME)
    parser.add_argument('--img-size', type=int, default=DEFAULT_IMGSZ)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS_LONG)
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_LARGE)
    args = parser.parse_args()

    if args.version == 'v1':
        log.error('Check ultralytics/nn/modules/head/Detect line 36: self.reg_max=16 for TinyissimoYOLOv1.3')
        return

    from ultralytics import YOLO

    model = YOLO(
        f'./results/{args.exp_id}/weights/last.pt'
        if args.load
        else f'./ultralytics/cfg/models/tinyissimo/tinyissimo-{args.version}.yaml'
    )

    model.train(
        data='coco.yaml',
        project=DEFAULT_PROJECT,
        name=DEFAULT_EXP_NAME,
        optimizer=DEFAULT_SGD,
        imgsz=args.img_size,
        epochs=args.epochs,
        batch=args.batch,
    )

    model.export(format='onnx', project=DEFAULT_PROJECT, name=DEFAULT_EXP_NAME, imgsz=[args.img_size, args.img_size])


if __name__ == '__main__':
    main()
