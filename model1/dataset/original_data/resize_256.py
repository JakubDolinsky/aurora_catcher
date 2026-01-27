from pathlib import Path
from shutil import copy2  # použijeme copy2 namiesto move
from PIL import Image

# ================= configuration =================
SPLITS = ["train", "validation", "test"]
TARGET_LONG_EDGE = 256
IMAGE_EXTENSIONS = (".jpg", ".jpeg")
LABELS_PREFIX = "labels"
SUFFIX = "_256"
# ================================================

def main():
    for split in SPLITS:
        input_dir = Path(split)
        output_dir = Path(f"{split}{SUFFIX}")

        if not input_dir.exists():
            print(f"Skipping '{split}': directory does not exist.")
            continue

        output_dir.mkdir(exist_ok=True)
        print(f"Processing split '{split}' -> '{output_dir}'")

        original_labels = input_dir / f"{LABELS_PREFIX}_{split}.csv"
        if original_labels.exists():
            new_labels_name = f"{LABELS_PREFIX}_{split}.csv"
            copy2(original_labels, output_dir / new_labels_name)
            print(f"Copied and renamed labels CSV to '{output_dir / new_labels_name}'")

        # count of processed images
        processed_count = 0

        for file_path in input_dir.iterdir():
            if not file_path.suffix.lower() in IMAGE_EXTENSIONS:
                continue

            dst_file = output_dir / file_path.name

            with Image.open(file_path) as img:
                img = img.convert("RGB")
                width, height = img.size

                # scale by longer side
                scale = TARGET_LONG_EDGE / max(width, height)
                new_width = int(round(width * scale))
                new_height = int(round(height * scale))

                resized_img = img.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
                resized_img.save(dst_file, quality=95)

            processed_count += 1

        print(f"Processed {processed_count} images in '{split}' -> '{output_dir}'\n")

    print("Done! All pictures has been copied and resized. Also csv file has been copied.")
