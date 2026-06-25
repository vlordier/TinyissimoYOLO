import argparse

import tinyissimo_yolo._vendored  # noqa: F401  ensure vendored ultralytics is on sys.path


def main():
    parser = argparse.ArgumentParser(description='Train and export a TinyissimoYOLO model')
    parser.add_argument('--version', default='v8', choices=['v1', 'v8'])
    parser.add_argument('--load', action='store_true', help='Load existing weights')
    parser.add_argument('--exp-id', default='exp1')
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=512)
    args = parser.parse_args()

    if args.version == 'v1':
        print('Please, check to modify ultralytics/nn/modules/head/Detect')
        print('for TinyissimoYOLOv1.3 small and big change')
        print('line 36 to: self.reg_max=16')
        return

    from ultralytics import YOLO

    if args.load:
        model_name = f'./results/{args.exp_id}/weights/last.pt'
        model = YOLO(model_name)
    else:
        model_name = f'./ultralytics/cfg/models/tinyissimo/tinyissimo-{args.version}.yaml'
        model = YOLO(model_name)

    model.train(
        data='coco.yaml',
        project='results',
        name='exp',
        optimizer='SGD',
        imgsz=args.img_size,
        epochs=args.epochs,
        batch=args.batch,
    )

    model.export(format='onnx', project='results', name='exp', imgsz=[args.img_size, args.img_size])


if __name__ == '__main__':
    main()
