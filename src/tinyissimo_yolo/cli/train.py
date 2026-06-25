import argparse

import tinyissimo_yolo._vendored  # noqa: F401  ensure vendored ultralytics is on sys.path


def main():
    parser = argparse.ArgumentParser(description='Train a TinyissimoYOLO model with tiling')
    parser.add_argument('--tiling-config', default='tiling_config.yaml')
    parser.add_argument('--data', default='CARPK_tiling.yaml')
    parser.add_argument('--model', default='tinyissimo-v1-small.yaml')
    parser.add_argument('--imgsz', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--project', default=None)
    args = parser.parse_args()

    import wandb

    from ultralytics import YOLO
    from ultralytics.utils.offline_tiling import Tiler

    wandb.init(project=args.project or 'ultralytics-test')

    tiler = Tiler(args.tiling_config)
    tiler.get_split_dataset()

    model = YOLO(args.model)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, single_cls=True)

    num_layers = sum(1 for _ in model.model.model.modules()) - 1
    print(f'Number of layers: {num_layers}')

    num_params = sum(p.numel() for p in model.model.model.parameters())
    print(f'Number of parameters: {num_params}')

    model.export(format='onnx', imgsz=[args.imgsz, args.imgsz], opset=12)


if __name__ == '__main__':
    main()
