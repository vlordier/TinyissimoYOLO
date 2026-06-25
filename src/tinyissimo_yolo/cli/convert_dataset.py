import argparse

from tinyissimo_yolo.utils.dataset import convert_carpk_to_create_ml, convert_create_ml_to_yolo


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert CARPK dataset to YOLO format', add_help=False)
    parser.add_argument('--images-dir', default='../../datasets/CARPK_devkit/data/Images')
    parser.add_argument('--labels-dir', default='../../datasets/CARPK_devkit/data/Annotations')
    parser.add_argument('--new-data-dir', default='../../datasets/CARPK')
    return parser


def main() -> None:
    parser = get_parser()
    parser.add_help = True
    args = parser.parse_args()
    labels = convert_carpk_to_create_ml(args.labels_dir, args.images_dir)
    convert_create_ml_to_yolo(labels, args.images_dir, args.new_data_dir)


if __name__ == '__main__':
    main()
