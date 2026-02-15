from pathlib import Path

import torch

#configuration for 256x256 and 128x128 pictures
PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_ROOT = PROJECT_ROOT / "dataset"
CLASSES_ROOT_DIR = DATASET_ROOT / "original_data"
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
    return dataset_dir


def get_labels_csv(dataset_dir: Path) -> Path:
    dataset_root_dir = DATASET_ROOT

    folder_name = dataset_dir.name                # e.g. "train_256"
    labels_suffix = folder_name.split("_")[0]     # "train"

    csv_path = dataset_root_dir / f"labels_{labels_suffix}.csv"
    return csv_path


TRAIN_256_DIR = get_dataset_dir("train", DATASET_NAMES_256)
TRAIN_CSV = get_labels_csv(TRAIN_256_DIR)
VAL_256_DIR = get_dataset_dir("val", DATASET_NAMES_256)
VAL_CSV = get_labels_csv(VAL_256_DIR)
TEST_256_DIR = get_dataset_dir("test", DATASET_NAMES_256)
TEST_CSV = get_labels_csv(TEST_256_DIR)
HARD_VALIDATION_256_DIR = get_dataset_dir("hard_validation", DATASET_NAMES_256)
HARD_VALIDATION_256_CSV = get_labels_csv(HARD_VALIDATION_256_DIR)

TRAIN_128_DIR = get_dataset_dir("train", DATASET_NAMES_128)
VAL_128_DIR = get_dataset_dir("val", DATASET_NAMES_128)

CLASS_NAMES = ["airglow", "light pollution", "lightning", "milky way", "NLC", "twilight", "zodiacal light"]
#set true if model is been training on 128*128 dataset
IS_128x128  = False

#hyperparameters
BATCH_SIZE = 16
EPOCHS =100
LR = 0.0003
WEIGHT_DECAY = 0.0001
#for 256*256 training
SCHEDULER_PATIENCE = 6
PATIENCE = 12
#for 128*128 training
#SCHEDULER_PATIENCE = 6
#PATIENCE = 5

#thresholds obtained in script adjust_positive_threshold_and_prob_levels.py
DECISION_F1_THRESHOLDS = torch.tensor([
    0.3600,  # airglow
    0.3200,  # light pollution
    0.3800,  # lightning
    0.5000,  # milky way
    0.4600,  # NLC
    0.2600,  # twilight
    0.4100,  # zodiacal light
], dtype=torch.float32)

#probability points for low, medium and high probability obtained in script adjust_positive_threshold_and_prob_levels.py
DECISION_PROBABILITY_LEVELS = torch.tensor([
    [0.3600, 0.9342, 0.9795],  # airglow
    [0.3200, 0.8194, 0.9368],  # light pollution
    [0.3800, 0.9864, 0.9971],  # lightning
    [0.5000, 0.9193, 0.9692],  # milky way
    [0.4600, 0.9961, 0.9984],  # NLC
    [0.2600, 0.9767, 0.9942],  # twilight
    [0.4100, 0.7018, 0.8013],  # zodiacal light
], dtype=torch.float32)

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
ROTATION_ANGLE = 5
ROTATION_P = 0.2

#128x128 (BRIGHTNESS = 0.12, CONTRAST = 0.12, COLOR_JITTER_P = 0.2)
#256x256 (BRIGHTNESS = 0.12, CONTRAST = 0.12, COLOR_JITTER_P = 0.3)
BRIGHTNESS = 0.12
CONTRAST = 0.12
COLOR_JITTER_P = 0.3

#256x256 (GAUSSIAN_NOISE = 0.005, GAUSSIAN_NOISE_P = 0.2)
GAUSSIAN_NOISE = 0.005
GAUSSIAN_NOISE_P = 0.2





