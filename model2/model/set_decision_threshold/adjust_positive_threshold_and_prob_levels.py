import datetime
import os
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from model2.model.cnn_model import AuroraCNN
from model2.model.aurora_dataset import validation_loader
from model2 import config

def log(msg, file=None):
    print(msg)
    if file is not None:
        file.write(msg + "\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "best_model_pt", "best_model.pt")

result_dir = os.path.join(base_dir, "result")
os.makedirs(result_dir, exist_ok=True)
run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(result_dir, f"validation_thresholds_and_metrics_{run_id}.txt")

# --------------------
# LOAD MODEL
# --------------------
model = AuroraCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Loss criterion
criterion = torch.nn.BCEWithLogitsLoss()

# --------------------
# COLLECT PREDICTIONS & LOSS
# --------------------
all_probs = []
all_targets = []
all_losses = []

with torch.inference_mode():
    for imgs, labels in validation_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)

        # loss pre batch
        batch_loss = criterion(logits, labels)
        all_losses.append(batch_loss.item())

        # probabilities for metrics
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_targets.append(labels.cpu())

probs = torch.cat(all_probs).numpy()  # [N, C]
targets = torch.cat(all_targets).numpy()  # [N, C]
loss_val = np.mean(all_losses)  # priemer cez batch-e

num_classes = len(config.CLASS_NAMES)
threshold_grid = np.linspace(0.05, 0.95, 91)

# --------------------
# THRESHOLD SEARCH + KVANTILY
# --------------------
f1_thresholds = np.zeros(num_classes)
low_q = np.zeros(num_classes)
med_q = np.zeros(num_classes)
high_q = np.zeros(num_classes)

for c in range(num_classes):
    y_true = targets[:, c]
    y_prob = probs[:, c]

    # F1-optimal threshold
    best_f1 = -1
    best_t = 0.5
    for t in threshold_grid:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    f1_thresholds[c] = best_t

    # Quantils for low/medium/high above threshold
    mask = y_prob >= best_t
    probs_above = y_prob[mask]
    if len(probs_above) > 0:
        low_q[c] = best_t  # spodná hranica low level
        med_q[c] = np.quantile(probs_above, 0.5)  # médium = 50. percentil
        high_q[c] = np.quantile(probs_above, 0.66)  # high = 66. percentil
    else:
        # fallback, if ni probability above threshold
        low_q[c] = med_q[c] = high_q[c] = best_t

# --------------------
# VALIDATION METRICS
# --------------------
# Binarize predictions according to F1 threshold
y_pred_matrix = np.zeros_like(probs, dtype=int)
for c in range(num_classes):
    y_pred_matrix[:, c] = (probs[:, c] >= f1_thresholds[c]).astype(int)

# Accuracy / F1 / precision / recall
accuracy_val = accuracy_score(targets.flatten(), y_pred_matrix.flatten())
f1_val = f1_score(targets, y_pred_matrix, average='macro', zero_division=0)
precision_val = precision_score(targets, y_pred_matrix, average='macro', zero_division=0)
recall_val = recall_score(targets, y_pred_matrix, average='macro', zero_division=0)

# --------------------
# FUNCTIONAL CHECK
# --------------------
per_sample_hit_ratios = []

for i in range(probs.shape[0]):
    true_classes = set(np.where(targets[i] == 1)[0])

    if len(true_classes) == 0:
        continue

    predicted_classes = set(
        c for c in range(num_classes)
        if probs[i, c] >= f1_thresholds[c]
    )

    correctly_predicted = true_classes & predicted_classes

    hit_ratio = len(correctly_predicted) / len(true_classes)
    per_sample_hit_ratios.append(hit_ratio)

per_sample_hit_ratios = np.array(per_sample_hit_ratios)
avg_hit_ratio = per_sample_hit_ratios.mean()

# --------------------
# PRINT RESULTS
# --------------------
with open(log_file_path, 'w') as f:

    log("F1-optimal thresholds per class:", f)
    for c in range(num_classes):
        log(
            f"{config.CLASS_NAMES[c]:<15}: "
            f"{f1_thresholds[c]:.3f}, "
            f"Low={low_q[c]:.3f}, "
            f"Med={med_q[c]:.3f}, "
            f"High={high_q[c]:.3f}",
            f
        )

    log("\nValidation metrics:", f)
    log(
        f"Loss={loss_val:.4f}, "
        f"Accuracy={accuracy_val:.3f}, "
        f"F1={f1_val:.3f}, "
        f"Precision={precision_val:.3f}, "
        f"Recall={recall_val:.3f}",
        f
    )

    log(f"Avg. per-sample hit ratio={avg_hit_ratio:.3f}", f)

    f.write("\n\n# ===== COPY TO config.py =====\n\n")

    f.write("DECISION_F1_THRESHOLDS = torch.tensor([\n")
    for c in range(num_classes):
        f.write(f"    {f1_thresholds[c]:.4f},  # {config.CLASS_NAMES[c]}\n")
    f.write("], dtype=torch.float32)\n\n")

    f.write("DECISION_PROBABILITY_LEVELS = torch.tensor([\n")
    for c in range(num_classes):
        f.write(
            f"    [{low_q[c]:.4f}, {med_q[c]:.4f}, {high_q[c]:.4f}],  "
            f"# {config.CLASS_NAMES[c]}\n"
        )
    f.write("], dtype=torch.float32)\n")
