"""Shared constants for TinyissimoYOLO."""

# ---------------------------------------------------------------------------
# Drawing / visualization
# ---------------------------------------------------------------------------
COLOR_RED: tuple[int, int, int] = (255, 0, 0)
COLOR_GREEN: tuple[int, int, int] = (0, 255, 0)
COLOR_BLUE: tuple[int, int, int] = (0, 0, 255)

RECT_THIN: int = 1
RECT_NORMAL: int = 2
RECT_BOLD: int = 3
RECT_THICK: int = 4
CIRCLE_RADIUS: int = 5
CIRCLE_FILL: int = -1

# ---------------------------------------------------------------------------
# Image / model defaults
# ---------------------------------------------------------------------------
DEFAULT_IMGSZ: int = 256
DEFAULT_OPSET: int = 12
IMAGE_EXTENSIONS: tuple[str, ...] = ('.jpg', '.png')
LABEL_EXTENSION: str = '.txt'
TILE_SEPARATOR: str = '_'

# ---------------------------------------------------------------------------
# Dataset / annotation
# ---------------------------------------------------------------------------
CAR_CLASS_ID: int = 0
CAR_LABEL: str = 'car'
YOLO_ROUND_DECIMALS: int = 6

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESH: float = 0.6
DEFAULT_IOU_THRESH: float = 0.5
IOU_SWEEP_START: float = 0.5
IOU_SWEEP_END: float = 0.95
IOU_SWEEP_STEPS: int = 10

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------
CARPK_FOLDERS: tuple[str, str, str] = ('CARPK_train', 'CARPK_val', 'CARPK_test')
SPLIT_URLS: dict[str, str] = {
    'train': 'https://github.com/mojulian/ultralytics/releases/download/0.1/train_images.txt',
    'val': 'https://github.com/mojulian/ultralytics/releases/download/0.1/val_images.txt',
    'test': 'https://github.com/mojulian/ultralytics/releases/download/0.1/test.txt',
}

# ---------------------------------------------------------------------------
# Paths / config files
# ---------------------------------------------------------------------------
TILING_CONFIG: str = 'tiling_config.yaml'
DATASET_YAML: str = 'ultralytics/cfg/datasets/CARPK_tiling.yaml'
MODEL_YAML_TINYISSIMO: str = 'tinyissimo-v1-small.yaml'

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS_SHORT: int = 1
DEFAULT_EPOCHS_LONG: int = 1000
DEFAULT_BATCH_SMALL: int = 64
DEFAULT_BATCH_LARGE: int = 512
DEFAULT_PROJECT: str = 'results'
DEFAULT_EXP_NAME: str = 'exp'
DEFAULT_SGD: str = 'SGD'
