"""Shared constants for TinyissimoYOLO."""

# ---------------------------------------------------------------------------
# Drawing / visualization
# ---------------------------------------------------------------------------
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)

RECT_THIN = 1
RECT_NORMAL = 2
RECT_BOLD = 3
RECT_THICK = 4
CIRCLE_RADIUS = 5
CIRCLE_FILL = -1

# ---------------------------------------------------------------------------
# Image / model defaults
# ---------------------------------------------------------------------------
DEFAULT_IMGSZ = 256
DEFAULT_OPSET = 12
IMAGE_EXTENSIONS = ('.jpg', '.png')
LABEL_EXTENSION = '.txt'
TILE_SEPARATOR = '_'

# ---------------------------------------------------------------------------
# Dataset / annotation
# ---------------------------------------------------------------------------
CAR_CLASS_ID = 0
CAR_LABEL = 'car'
YOLO_ROUND_DECIMALS = 6
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESH = 0.6
DEFAULT_IOU_THRESH = 0.5
IOU_SWEEP_START = 0.5
IOU_SWEEP_END = 0.95
IOU_SWEEP_STEPS = 10

# ---------------------------------------------------------------------------
# Dataset splits
# ---------------------------------------------------------------------------
CARPK_FOLDERS = ('CARPK_train', 'CARPK_val', 'CARPK_test')
SPLIT_URLS = {
    'train': 'https://github.com/mojulian/ultralytics/releases/download/0.1/train_images.txt',
    'val': 'https://github.com/mojulian/ultralytics/releases/download/0.1/val_images.txt',
    'test': 'https://github.com/mojulian/ultralytics/releases/download/0.1/test.txt',
}

# ---------------------------------------------------------------------------
# Paths / config files
# ---------------------------------------------------------------------------
TILING_CONFIG = 'tiling_config.yaml'
DATASET_YAML = 'ultralytics/cfg/datasets/CARPK_tiling.yaml'
MODEL_YAML_TINYISSIMO = 'tinyissimo-v1-small.yaml'

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS_SHORT = 1
DEFAULT_EPOCHS_LONG = 1000
DEFAULT_BATCH_SMALL = 64
DEFAULT_BATCH_LARGE = 512
DEFAULT_PROJECT = 'results'
DEFAULT_EXP_NAME = 'exp'
DEFAULT_SGD = 'SGD'
