import argparse

import tinyissimo_yolo._vendored  # noqa: F401
from tinyissimo_yolo._constants import (
    DEFAULT_BATCH_SMALL,
    DEFAULT_EPOCHS_SHORT,
    DEFAULT_IMGSZ,
    DEFAULT_OPSET,
    MODEL_YAML_TINYISSIMO,
    TILING_CONFIG,
)
from tinyissimo_yolo._logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a TinyissimoYOLO model with tiling')
    parser.add_argument('--tiling-config', default=TILING_CONFIG)
    parser.add_argument('--data', default='CARPK_tiling.yaml')
    parser.add_argument('--model', default=MODEL_YAML_TINYISSIMO)
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ)
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS_SHORT)
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SMALL)
    parser.add_argument('--project', default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--single-cls', action='store_true', default=True)
    args = parser.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils.offline_tiling import Tiler

    if not args.no_wandb:
        try:
            import wandb

            wandb.init(project=args.project or 'ultralytics-test')
        except Exception as exc:
            log.warning('wandb init skipped (%s)', exc)

    Tiler(args.tiling_config).get_split_dataset()

    model = YOLO(args.model)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, single_cls=args.single_cls)

    n_layers = sum(1 for _ in model.model.model.modules()) - 1
    n_params = sum(p.numel() for p in model.model.model.parameters())
    log.info('Layers: %d, Parameters: %d', n_layers, n_params)

    model.export(format='onnx', imgsz=[args.imgsz, args.imgsz], opset=DEFAULT_OPSET)


if __name__ == '__main__':
    main()
