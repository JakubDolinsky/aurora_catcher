import csv
import random
from pathlib import Path
from shutil import copy2  # namiesto move

# ================== configuration ==================
AURORA_DIR = Path("aurora")
NON_AURORA_DIR = Path("non_aurora")

SPLITS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
START_INDEX = 1
SEED = 42
# ==================================================

random.seed(SEED)


def collect_images(folder: Path, label: int):
    images = []
    for f in folder.iterdir():
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append({"path": f, "label": label})
    return images


def split_dataset(images):
    random.shuffle(images)
    n = len(images)

    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["validation"])

    return {
        "train": images[:n_train],
        "validation": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:]
    }


def ensure_dirs():
    for split in SPLITS.keys():
        Path(split).mkdir(exist_ok=True)


def main():
    ensure_dirs()

    # load datasets
    aurora = collect_images(AURORA_DIR, label=1)
    non_aurora = collect_images(NON_AURORA_DIR, label=0)

    # split every dataset separately
    aurora_split = split_dataset(aurora)
    non_aurora_split = split_dataset(non_aurora)

    # join and shuffle for each split
    final_splits = {}
    for split in SPLITS.keys():
        combined = aurora_split[split] + non_aurora_split[split]
        random.shuffle(combined)
        final_splits[split] = combined

    # rename + copy + create csv file
    img_index = START_INDEX

    for split, items in final_splits.items():
        split_dir = Path(split)
        csv_path = split_dir / f"labels_{split}.csv"

        csv_rows = []
        for item in items:
            src = item["path"]
            label = item["label"]

            new_name = f"img_{img_index:06d}{src.suffix.lower()}"
            dst = split_dir / new_name

            copy2(src, dst)
            csv_rows.append([new_name, label])
            img_index += 1

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])
            writer.writerows(csv_rows)

    print("Done!")
    print("Datasets are splitted, copied and labelled.")


if __name__ == "__main__":
    main()
