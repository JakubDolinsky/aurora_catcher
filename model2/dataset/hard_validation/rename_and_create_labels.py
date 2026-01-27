import os
import csv

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
CSV_FILENAME = "labels.csv"
START_INDEX = 1
ZERO_PADDING = 4  # img_0001.jpg
# ---------------------

root_dir = os.getcwd()

label_dirs = [
    d for d in sorted(os.listdir(root_dir))
    if os.path.isdir(d) and not d.startswith(".")
]

if not label_dirs:
    raise RuntimeError("No subdirectories found (labels).")

print("Labels found:", label_dirs)

# CSV header
header = ["filename"] + label_dirs

rows = []
counter = START_INDEX

for label in label_dirs:
    label_path = os.path.join(root_dir, label)

    for fname in sorted(os.listdir(label_path)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        new_name = f"img_{str(counter).zfill(ZERO_PADDING)}{ext}"
        old_path = os.path.join(label_path, fname)
        new_path = os.path.join(label_path, new_name)

        if fname != new_name:
            os.rename(old_path, new_path)

        # create label vector
        row = {"filename": new_name}
        for l in label_dirs:
            row[l] = 1 if l == label else 0

        rows.append(row)
        counter += 1

# write CSV
csv_path = os.path.join(root_dir, CSV_FILENAME)
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(f"Hotovo. Premenovaných obrázkov: {counter - START_INDEX}")
print(f"CSV uložené ako: {CSV_FILENAME}")
