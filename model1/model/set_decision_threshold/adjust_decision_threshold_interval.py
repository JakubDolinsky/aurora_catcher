import os
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model1 import config
from model1.model.cnn_model import AuroraCNN
from model1.model.aurora_dataset import validation_loader

# =========================
# FILES & DIRECTORIES
# =========================
result_dir = os.path.join(os.path.dirname(__file__), "result")
os.makedirs(result_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = os.path.join(result_dir, f"adjust_threshold_{timestamp}_log.txt")
probs_file_path = os.path.join(result_dir, f"all_probabilities_{timestamp}.txt")

base_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "best_model_pt", "best_model.pt")

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AuroraCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# =========================
# THRESHOLDS
# =========================
T_FALSE = config.SIGMOID_THRESHOLDS[0]
T_TRUE = config.SIGMOID_THRESHOLDS[1]

# =========================
# EVALUATION LOOP
# =========================
criterion = nn.BCEWithLogitsLoss()
all_labels = []
all_preds = []
all_states = []
all_probs = []
losses = []

print("Test for threshold decision intervals adjustment started")

with torch.inference_mode():
    for imgs, labels in validation_loader:
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(imgs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        probs = torch.sigmoid(outputs).squeeze(1)
        all_probs.extend(probs.cpu().numpy())
        labels_cpu = labels.squeeze(1).cpu().numpy()
        all_labels.extend(labels_cpu)

        for p, y in zip(probs, labels_cpu):
            if p <= T_FALSE:
                state = "false"
                all_preds.append(0)
            elif p >= T_TRUE:
                state = "true"
                all_preds.append(1)
            else:
                state = "uncertain"
                all_preds.append(None)
            all_states.append(state)

# =========================
# CONFUSION MATRIX
# =========================
TN = FP = FN = TP = 0
U_F = U_T = 0

for y, pred in zip(all_labels, all_preds):
    if pred is None:
        if y == 0:
            U_F += 1
        else:
            U_T += 1
    elif pred == 0:
        if y == 0:
            TN += 1
        else:
            FN += 1
    elif pred == 1:
        if y == 1:
            TP += 1
        else:
            FP += 1

uncertain_rate = (U_F + U_T) / len(all_labels)

print("===== INTERVAL CONFUSION MATRIX =====")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")
print(f"TP: {TP}")
print(f"Uncertain (False): {U_F}")
print(f"Uncertain (True):  {U_T}")
print(f"Total uncertain:   {U_F + U_T}")
print(f"Uncertain rate: {uncertain_rate:.2%}")

with open(log_file_path, "w", encoding="utf-8") as f:
    f.write("===== INTERVAL CONFUSION MATRIX =====\n")
    f.write(f"TN: {TN}\n")
    f.write(f"FP: {FP}\n")
    f.write(f"FN: {FN}\n")
    f.write(f"TP: {TP}\n")
    f.write(f"Uncertain (False): {U_F}\n")
    f.write(f"Uncertain (True):  {U_T}\n")
    f.write(f"Total uncertain:   {U_F + U_T}\n")
    f.write(f"Uncertain rate: {uncertain_rate:.2%}\n")

print(f"Results saved to: {log_file_path}")

# =========================
# VISUALIZATION
# =========================
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

probs_false = all_probs[all_labels == 0]
probs_true = all_probs[all_labels == 1]

plt.figure(figsize=(10,6))
sns.histplot(probs_false, color="red", label="False", bins=50, kde=True, stat="density", alpha=0.5)
sns.histplot(probs_true, color="green", label="True", bins=50, kde=True, stat="density", alpha=0.5)

plt.axvline(T_FALSE, color="red", linestyle="--", label="T_FALSE")
plt.axvline(T_TRUE, color="green", linestyle="--", label="T_TRUE")
plt.title("Probability distribution by true label")
plt.xlabel("Predicted probability")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

# Save plot
plot_path = os.path.join(result_dir, f"probability_distribution_{timestamp}.png")
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"Probability distribution plot saved to: {plot_path}")
