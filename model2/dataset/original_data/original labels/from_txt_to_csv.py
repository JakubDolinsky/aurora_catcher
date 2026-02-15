import csv
import pandas as pd

SHORTCUT_TO_COLUMN = {
    "a": "airglow",
    "lp": "light pollution",
    "l": "lightning",
    "mw": "milky way",
    "nlc": "NLC",
    "t": "twilight",
    "zl": "zodiacal light",
}

CSV_COLUMNS = list(SHORTCUT_TO_COLUMN.values())

INPUT_TXT = "validation_label_text.txt"
INPUT_CSV = "labels_validation.csv"
OUTPUT_CSV = "labels_validation_from_txt.csv"

df = pd.read_csv(INPUT_CSV)

for col in CSV_COLUMNS:
    if col not in df.columns:
        raise ValueError(f"Column is missing in CSV: {col}")

with open(INPUT_TXT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = [p for p in line.split(",") if p]
        filename = parts[0]
        labels = parts[1:]

        mask = df["filename"] == filename
        if not mask.any():
            continue

        df.loc[mask, CSV_COLUMNS] = 0

        for lbl in labels:
            if lbl == "unknown":
                continue

            if lbl not in SHORTCUT_TO_COLUMN:
                raise ValueError(f"Unknown abbrevation in TXT: {lbl}")

            col_name = SHORTCUT_TO_COLUMN[lbl]
            df.loc[mask, col_name] = 1

if (df[CSV_COLUMNS] == -1).any().any():
    raise RuntimeError("CSV still contains -1 which is forbidden.")


df.to_csv(OUTPUT_CSV, index=False)
print("Done. CSV has been created without -1.")
