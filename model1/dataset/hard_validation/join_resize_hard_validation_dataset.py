import os
import random
from PIL import Image

# --- CONFIG ---
SUBFOLDERS = ["aurora", "non_aurora"]
DEST_FOLDER_NAME = "hard_validation_256"
LONG_SIDE = 256
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg",}

parent_dir = os.getcwd()

destination_folder = os.path.join(os.path.dirname(parent_dir), DEST_FOLDER_NAME)
os.makedirs(destination_folder, exist_ok=True)

all_images = []

for sub in SUBFOLDERS:
    sub_path = os.path.join(parent_dir, sub)
    if not os.path.exists(sub_path):
        raise RuntimeError(f"Subfolder {sub_path} does not exist")
    for fname in os.listdir(sub_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            all_images.append(os.path.join(sub_path, fname))

random.seed(SEED)
random.shuffle(all_images)

print(f"Total images: {len(all_images)}")

for index, img_path in enumerate(all_images, start=1):
    with Image.open(img_path) as img:
        w, h = img.size
        if w >= h:
            new_w = LONG_SIDE
            new_h = int(LONG_SIDE * h / w)
        else:
            new_h = LONG_SIDE
            new_w = int(LONG_SIDE * w / h)

        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        ext = os.path.splitext(img_path)[1].lower()
        original_name = os.path.basename(img_path)
        dest_path = os.path.join(destination_folder, original_name)
        img_resized.save(dest_path)

        print(f"{original_name} ({new_w}x{new_h})")

print("Done. Resized files have been saved into:", destination_folder)
