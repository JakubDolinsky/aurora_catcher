import os
import random
import shutil
from pathlib import Path
from PIL import Image
import csv

from model2 import config

# =====================
# CONFIG
# =====================
DATASET_ROOT = config.DATASET_ROOT
CLASSES_ROOT_DIR = config.CLASSES_ROOT_DIR
OUTPUT_DIRS = {
    "train_256": config.TRAIN_256_DIR,
    "train_128": config.TRAIN_128_DIR,
    "validation_256": config.VAL_256_DIR,
    "validation_128": config.VAL_128_DIR,
    "test_256": config.TEST_256_DIR
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# =====================
# HELPERS
# =====================

def resize_and_save(src, dst, long_side):
    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = long_side / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.BICUBIC)
    img.save(dst, quality=95)

def shuffle_files_in_directory(dir_path, extensions={".jpg", ".jpeg", ".png"}):
    dir_path = Path(dir_path)
    files = [f for f in dir_path.iterdir() if f.suffix.lower() in extensions]

    if len(files) <= 1:
        return

    # náhodné poradie
    shuffled = files[:]
    random.shuffle(shuffled)

    # 1️⃣ dočasné názvy
    temp_names = []
    for i, f in enumerate(shuffled):
        tmp = f.with_name(f"__tmp__{i}{f.suffix}")
        f.rename(tmp)
        temp_names.append(tmp)

    # 2️⃣ späť na pôvodné názvy, ale v inom poradí
    for tmp, original in zip(temp_names, files):
        tmp.rename(dir_path / original.name)

# =====================
# MAIN
# =====================

def main():
    classes = sorted([
        d.name for d in CLASSES_ROOT_DIR.iterdir()
        if d.is_dir() and not d.name.endswith(("256", "128"))
    ])

    print("Classes:", classes)

    for d in OUTPUT_DIRS.values():
        d.mkdir(exist_ok=True)

    labels = {
        "train": [],
        "validation": [],
        "test": []
    }

    global_idx = 1

    for class_name in classes:
        class_dir = CLASSES_ROOT_DIR / class_name
        images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, files in splits.items():
            for src in files:
                ext = src.suffix.lower()
                name = f"img_{global_idx:04d}{ext}"
                global_idx += 1

                # prepare label row
                row = {"filename": name}
                for c in classes:
                    row[c] = 1 if c == class_name else -1
                labels[split].append(row)

                if split == "train":
                    resize_and_save(src, OUTPUT_DIRS["train_256"] / name, 256)
                    resize_and_save(src, OUTPUT_DIRS["train_128"] / name, 128)

                elif split == "validation":
                    resize_and_save(src, OUTPUT_DIRS["validation_256"] / name, 256)
                    resize_and_save(src, OUTPUT_DIRS["validation_128"] / name, 128)

                elif split == "test":
                    resize_and_save(src, OUTPUT_DIRS["test_256"] / name, 256)

    # =====================
    # SAVE CSVs
    # =====================

    for split in ["train", "validation", "test"]:
        csv_path =DATASET_ROOT / f"labels_{split}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename"] + classes
            )
            writer.writeheader()
            writer.writerows(labels[split])

        print(f"Saved {csv_path}")

    print("DONE.")


if __name__ == "__main__":
    main()
