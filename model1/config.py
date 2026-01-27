from pathlib import Path

#configuration for 256x256 and 128x128 pictures
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_ROOT = PROJECT_ROOT / "dataset"
DATASET_NAMES_256 = {
    "train": "train_256",
    "val": "validation_256",
    "test": "test_256",
    "hard_validation": "hard_validation_256"
}

DATASET_NAMES_128 = {
    "train": "train_128",
    "val": "validation_128"
}

# =====================================================
# HELPERS
# =====================================================
def get_dataset_dir(name: str, dataset_names) -> Path:
    dataset_dir = DATASET_ROOT / dataset_names[name]
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    return dataset_dir


def get_labels_csv(dataset_dir: Path) -> Path:
    dataset_dir = Path(dataset_dir)

    folder_name = dataset_dir.name                # e.g. "train_256"
    labels_suffix = folder_name.split("_")[0]     # "train"

    csv_path = dataset_dir / f"labels_{labels_suffix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {csv_path}")

    return csv_path


TRAIN_256_DIR = get_dataset_dir("train", DATASET_NAMES_256)
TRAIN_256_CSV = get_labels_csv(TRAIN_256_DIR)
VAL_256_DIR = get_dataset_dir("val", DATASET_NAMES_256)
VAL_256_CSV = get_labels_csv(VAL_256_DIR)
TEST_256_DIR = get_dataset_dir("test", DATASET_NAMES_256)
TEST_256_CSV = get_labels_csv(TEST_256_DIR)
HARD_VALIDATION_256_DIR = get_dataset_dir("hard_validation", DATASET_NAMES_256)
HARD_VALIDATION_256_CSV = get_labels_csv(HARD_VALIDATION_256_DIR)

TRAIN_128_DIR = get_dataset_dir("train", DATASET_NAMES_128)
TRAIN_128_CSV = get_labels_csv(TRAIN_128_DIR)
VAL_128_DIR = get_dataset_dir("val", DATASET_NAMES_128)
VAL_128_CSV = get_labels_csv(VAL_128_DIR)

#set true if model is been training on 128*128 dataset
IS_128x128  = False

#hyperparameters
BATCH_SIZE = 16
EPOCHS = 80
LR = 0.0003
WEIGHT_DECAY = 0.0001
#for 256*256 training
SCHEDULER_PATIENCE = 6
PATIENCE = 12
#for 128*128 training
#SCHEDULER_PATIENCE = 6
#PATIENCE = 5
SIGMOID_THRESHOLDS = (0.15, 0.75)
SIGMOID_THRESHOLD_TEST = 0.4125

#augmentation
#128x128 False
#256x256 True
ROTATION_APPLY = True
#128x128 True
#256x256 True
COLOR_JITTER_APPLY = True
#128x128 False
#256x256 True
GAUSSIAN_NOISE_APPLY = True

#set P values of each used augmentation to 1 while using it in unit tests
#256x266(ROTATION_ANGLE = 4, ROTATION_P = 0.2)
ROTATION_ANGLE = 5
ROTATION_P = 0.2

#128x128 (BRIGHTNESS = 0.12, CONTRAST = 0.12, COLOR_JITTER_P = 0.2)
#256x256 (BRIGHTNESS = 0.18, CONTRAST = 0.18, COLOR_JITTER_P = 0.3)
BRIGHTNESS = 0.12
CONTRAST = 0.12
COLOR_JITTER_P = 0.3

#256x256 (GAUSSIAN_NOISE = 0.005, GAUSSIAN_NOISE_P = 0.2)
GAUSSIAN_NOISE = 0.005
GAUSSIAN_NOISE_P = 0.2





