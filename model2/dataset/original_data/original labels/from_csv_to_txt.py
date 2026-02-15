import csv

PHENOMENA_MAP = {
    "airglow": "a",
    "light pollution": "lp",
    "lightning": "l",
    "milky way": "mw",
    "NLC": "nlc",
    "twilight": "t",
    "zodiacal light": "zl"
}

INPUT_CSV = "labels_train.csv"
OUTPUT_TXT = "train_label_text.txt"

with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as outfile:
        for row in reader:
            image_name = row[reader.fieldnames[0]]

            detected = []
            for phenomenon, shortcut in PHENOMENA_MAP.items():
                value = row.get(phenomenon)
                if value is not None and value.strip() == "1":
                    detected.append(shortcut)
                elif value.strip() == "-1":
                    detected.append("unknown")

            line = image_name
            if detected:
                line += "," + ",".join(detected) + ","

            outfile.write(line + "\n")
print("done")
