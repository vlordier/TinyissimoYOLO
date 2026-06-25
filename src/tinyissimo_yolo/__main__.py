"""Umbrella CLI: tinyissimo {train,evaluate,convert,export}."""

import argparse

COMMANDS = {
    'train': ('tinyissimo_yolo.cli.train', 'Train model with tiling'),
    'evaluate': ('tinyissimo_yolo.cli.evaluate', 'Evaluate model on tiled or full images'),
    'convert': ('tinyissimo_yolo.cli.convert_dataset', 'Convert CARPK dataset to YOLO format'),
    'export': ('tinyissimo_yolo.cli.train_export', 'Train and export model'),
}


def main() -> None:
    parser = argparse.ArgumentParser(description='TinyissimoYOLO — ultra-lightweight YOLO for edge deployment')
    sub = parser.add_subparsers(dest='command', required=True)

    for name, (module_path, help_text) in COMMANDS.items():
        import importlib

        mod = importlib.import_module(module_path)
        sub_parser = sub.add_parser(name, parents=[mod.get_parser()], help=help_text, add_help=False)
        # Re-add --help for the subparser (removed by parents=)
        sub_parser.add_argument('-h', '--help', action='help', help=argparse.SUPPRESS)

    args = parser.parse_args()

    mod = importlib.import_module(COMMANDS[args.command][0])
    mod.main()


if __name__ == '__main__':
    main()
