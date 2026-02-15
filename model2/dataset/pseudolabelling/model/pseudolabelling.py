import torch
import pandas as pd
from tqdm import tqdm
from model2.model.aurora_dataset import train_dataset, train_loader
from model2.model.cnn_model import AuroraCNN
from model2 import config

# -------- CONFIG --------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.BATCH_SIZE
PRINT_CONFIDENCE = True
CSV_PATH = "labels_train.csv"

# upper and lower thresholds for each class
UPPER_THRESHOLDS = [0.53, 0.95, 0.75, 0.75, 0.60, 0.60, 0.75]
LOWER_THRESHOLDS = [0.40, 0.35, 0.45, 0.45, 0.40, 0.45, 0.45]

# -------- LOADING DATA AND MODELU --------
dataset = train_dataset
dataloader = train_loader

model = AuroraCNN().to(DEVICE)
model.load_state_dict(torch.load("model_for_labeling/best_model.pt", map_location=DEVICE))
model.eval()

df = pd.read_csv(CSV_PATH)

# -------- PSEUDO-LABELING --------
print("Pseudolabeling has started")

with torch.no_grad():
    for idx in tqdm(range(len(dataset)), desc="Pseudo-labeling"):
        img, _ = dataset[idx]
        filename = dataset.image_files[idx]

        img = img.unsqueeze(0).to(DEVICE)
        logits = model(img)
        probs = torch.sigmoid(logits)
        probs_list = probs.squeeze(0).tolist()

        csv_row = df.loc[df['filename'] == filename]
        if csv_row.empty:
            continue

        orig_label = csv_row[config.CLASS_NAMES].values[0].astype(float)
        new_label = orig_label.copy()

        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            val = orig_label[class_idx]
            prob = probs_list[class_idx]

            if val == -1:
                if prob >= UPPER_THRESHOLDS[class_idx]:
                    new_label[class_idx] = 1
                elif prob <= LOWER_THRESHOLDS[class_idx]:
                    new_label[class_idx] = 0
                else:
                    new_label[class_idx] = -1

        df.loc[df['filename'] == filename, config.CLASS_NAMES] = new_label

        if PRINT_CONFIDENCE:
            conf_str = ", ".join([f"{cls}: {probs_list[j]:.3f}"
                                  for j, cls in enumerate(config.CLASS_NAMES)])
            print(f"{filename} | {conf_str}")

df.to_csv(CSV_PATH, index=False)
print("Pseudo-labeling dokončené. CSV aktualizované.")
