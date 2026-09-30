"""Project configuration for SVHN classification and detection."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Classification (cropped_digits) — edadaltocg/resnet18_svhn
# ---------------------------------------------------------------------------
MODEL_NAME = "resnet18_svhn"
MODEL_HUB_ID = "edadaltocg/resnet18_svhn"

DATASET_ID = "ufldl-stanford/svhn"
DATASET_CONFIG = "cropped_digits"
DATASET_SPLIT = "test"

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
INPUT_SIZE = 32

DEFAULT_BATCH_SIZE = 128
NUM_CLASSES = 10
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

# ---------------------------------------------------------------------------
# Detection + classification (full_numbers) — DETR
# ---------------------------------------------------------------------------
DETECTION_DATASET_CONFIG = "full_numbers"
DETECTION_TRAIN_SPLIT = "train"
DETECTION_TEST_SPLIT = "test"
DETECTION_OUTPUTS_DIR = OUTPUTS_DIR / "detection"

# COCO-style boxes in the HF card: [x, y, width, height]
BBOX_FORMAT = "xywh"

DETR_MODEL_ID = "facebook/detr-resnet-50"
DETR_IMAGE_SIZE = 480  # smaller than default 800 → faster subset training
DETR_BATCH_SIZE = 4
DETR_NUM_EPOCHS = 3
DETR_LEARNING_RATE = 1e-4
DETR_BACKBONE_LR_FACTOR = 0.1  # backbone trains slower than the new head
DETR_WEIGHT_DECAY = 1e-4
DETR_MAX_GRAD_NORM = 0.1  # DETR paper default; prevents NaN box explosions
DETR_DEFAULT_TRAIN_SAMPLES = 1000
DETR_DEFAULT_EVAL_SAMPLES = 200
ID2LABEL = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID = {name: i for i, name in ID2LABEL.items()}
