import torch
import numpy as np
import os

from model1.model.cnn_model import AuroraCNN
from model1.model.aurora_dataset import validation_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CREATE RESULT FOLDER
# =========================
result_dir = os.path.join(os.path.dirname(__file__), "result")
os.makedirs(result_dir, exist_ok=True)
log_file_path = os.path.join(result_dir, "validation_quantiles_log.txt")

base_dir = os.path.join(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "best_model_pt", "best_model.pt")

# =========================
# LOAD MODEL
# =========================
print("Load model")
model = AuroraCNN()
model.load_state_dict(
    torch.load(model_path, map_location=device)
)
model.to(device)
model.eval()

# =========================
# COLLECT PROBABILITIES
# =========================
probs_true = []
probs_false = []

with torch.inference_mode():
    print("Predict values in validation_dataset")
    for imgs, labels in validation_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs = torch.sigmoid(logits).squeeze(1)

        for p, y in zip(probs, labels):
            if y.item() == 1:
                probs_true.append(p.item())
            else:
                probs_false.append(p.item())

probs_true = np.array(probs_true)
probs_false = np.array(probs_false)

# =========================
# QUANTILES
# =========================
q_false_95 = np.quantile(probs_false, 0.95)
q_true_05 = np.quantile(probs_true, 0.05)

# =========================
# PREPARE LOG
# =========================
log_lines = [
    "===== VALIDATION QUANTILES =====",
    f"False class 95% quantile: {q_false_95:.4f}",
    f"True class  5% quantile: {q_true_05:.4f}"
]

if q_false_95 < q_true_05:
    log_lines.append("Clean separation gap detected")
    log_lines.append(f"Suggested uncertain zone: [{q_false_95:.4f}, {q_true_05:.4f}]")
else:
    log_lines.append("Overlap between false and true distributions")
    log_lines.append("Uncertain zone not cleanly separable")

# =========================
# PRINT AND SAVE LOG
# =========================
log_text = "\n".join(log_lines)
print(log_text)

with open(log_file_path, "w") as f:
    f.write(log_text)

print(f"\nLog saved to: {log_file_path}")
