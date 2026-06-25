"""Umbrella CLI: tinyissimo {train,evaluate,convert,export}."""

import argparse

from tinyissimo_yolo._logging import get_logger

log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description='TinyissimoYOLO — ultra-lightweight YOLO for edge deployment')
    sub = parser.add_subparsers(dest='command', required=True)

    p_train = sub.add_parser('train', help='Train model with tiling')
    p_train.add_argument('--tiling-config', default='tiling_config.yaml')
    p_train.add_argument('--data', default='CARPK_tiling.yaml')
    p_train.add_argument('--model', default='tinyissimo-v1-small.yaml')
    p_train.add_argument('--imgsz', type=int, default=256)
    p_train.add_argument('--epochs', type=int, default=1)
    p_train.add_argument('--batch', type=int, default=64)
    p_train.add_argument('--project', default=None)
    p_train.add_argument('--no-wandb', action='store_true')
    p_train.add_argument('--single-cls', action='store_true', default=True)

    p_eval = sub.add_parser('evaluate', help='Evaluate model on tiled or full images')
    p_eval.add_argument('--use-tiling', default=True, action=argparse.BooleanOptionalAction)
    p_eval.add_argument('--perform-iou-sweep', default=False, action=argparse.BooleanOptionalAction)
    p_eval.add_argument('--plot', default=False, action=argparse.BooleanOptionalAction)
    p_eval.add_argument('--image-set', default='test')
    p_eval.add_argument('--dataset-yaml-path', default='ultralytics/cfg/datasets/CARPK_tiling.yaml')
    p_eval.add_argument('--model-path', default='path/to/your/model.pt')
    p_eval.add_argument('--tiling-config', default='tiling_config.yaml')
    p_eval.add_argument('--conf-thresh', type=float, default=0.6)
    p_eval.add_argument('--iou-thresh', type=float, default=0.5)

    p_convert = sub.add_parser('convert', help='Convert CARPK dataset to YOLO format')
    p_convert.add_argument('--images-dir', default='../../datasets/CARPK_devkit/data/Images')
    p_convert.add_argument('--labels-dir', default='../../datasets/CARPK_devkit/data/Annotations')
    p_convert.add_argument('--new-data-dir', default='../../datasets/CARPK')

    p_export = sub.add_parser('export', help='Train and export model')
    p_export.add_argument('--version', default='v8', choices=['v1', 'v8'])
    p_export.add_argument('--load', action='store_true')
    p_export.add_argument('--exp-id', default='exp1')
    p_export.add_argument('--img-size', type=int, default=256)
    p_export.add_argument('--epochs', type=int, default=1000)
    p_export.add_argument('--batch', type=int, default=512)

    args = parser.parse_args()

    if args.command == 'train':
        from tinyissimo_yolo.cli.train import main as m
    elif args.command == 'evaluate':
        from tinyissimo_yolo.cli.evaluate import main as m
    elif args.command == 'convert':
        from tinyissimo_yolo.cli.convert_dataset import main as m
    elif args.command == 'export':
        from tinyissimo_yolo.cli.train_export import main as m

    m()


if __name__ == '__main__':
    main()
