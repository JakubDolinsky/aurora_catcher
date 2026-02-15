import datetime
import os
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from model2.model.aurora_dataset import test_loader
from model2.model.aurora_dataset import hard_validation_loader
from model2.model.cnn_model import AuroraCNN
from model2 import config
#set to test loader in case of running model on test dataset and use hard validation data loader in case of testing on hard validation dataset
#from model2.model.aurora_dataset import test_loader as data_loader
from model2.model.aurora_dataset import hard_validation_loader as data_loader


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
name_suffix =""
if data_loader is test_loader:
    name_suffix = "test"
elif data_loader is hard_validation_loader:
    name_suffix = "hard_validation"
log_file_path = os.path.join(result_dir, f"eval_metrics_log_{run_id}_{name_suffix}.txt")



# --------------------
# LOAD MODEL
# --------------------
model = AuroraCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --------------------
# COLLECT PREDICTIONS
# --------------------
all_probs = []
all_targets = []

with torch.inference_mode():
    for imgs, labels in data_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu()
        all_probs.append(probs)
        all_targets.append(labels)

probs = torch.cat(all_probs).numpy()      # [N, C]
targets = torch.cat(all_targets).numpy()  # [N, C]

num_classes = len(config.CLASS_NAMES)
f1_thresholds = config.DECISION_F1_THRESHOLDS.numpy()
low_level = config.DECISION_PROBABILITY_LEVELS[:,0].numpy()
med_level = config.DECISION_PROBABILITY_LEVELS[:,1].numpy()
high_level = config.DECISION_PROBABILITY_LEVELS[:,2].numpy()

# --------------------
# BINARIZE PREDICTIONS PODĽA F1 THRESHOLD
# --------------------
y_pred_matrix = np.zeros_like(probs, dtype=int)
for c in range(num_classes):
    y_pred_matrix[:, c] = (probs[:, c] >= f1_thresholds[c]).astype(int)

# --------------------
# LOSS + METRICS
# --------------------
criterion = torch.nn.BCELoss()
loss_val = criterion(torch.tensor(probs), torch.tensor(targets, dtype=torch.float32)).item()
accuracy_val = accuracy_score(targets.flatten(), y_pred_matrix.flatten())
f1_val = f1_score(targets, y_pred_matrix, average='macro', zero_division=0)
precision_val = precision_score(targets, y_pred_matrix, average='macro', zero_division=0)
recall_val = recall_score(targets, y_pred_matrix, average='macro', zero_division=0)

# --------------------
# PER-SAMPLE HIT RATIO
# --------------------
per_sample_hit_ratios = []
for i in range(probs.shape[0]):
    true_indices = np.where(targets[i] == 1)[0]
    pred_indices = np.where(probs[i] >= f1_thresholds)[0]
    if len(true_indices) > 0:
        hit_ratio = len(set(pred_indices) & set(true_indices)) / len(true_indices)
        per_sample_hit_ratios.append(hit_ratio)
per_sample_hit_ratios = np.array(per_sample_hit_ratios)
avg_hit_ratio = per_sample_hit_ratios.mean()

# --------------------
# LOG RESULTS
# --------------------
with open(log_file_path, 'w') as f:

    log("F1-optimal thresholds per class:", f)
    for c in range(num_classes):
        log(
            f"{config.CLASS_NAMES[c]:<15}: "
            f"{f1_thresholds[c]:.3f}, "
            f"Low={low_level[c]:.3f}, "
            f"Med={med_level[c]:.3f}, "
            f"High={high_level[c]:.3f}",
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
