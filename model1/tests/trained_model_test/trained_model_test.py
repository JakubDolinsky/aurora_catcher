import os
from datetime import (datetime)

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from torch.utils.data import DataLoader

from model1 import config
from model1.model.cnn_model import AuroraCNN
from model1.model.aurora_dataset import test_loader, build_transform, AuroraDataset

#set both variables to true in case of hard validation and threshold intervals
is_hard_validation_test = False
is_decision_threshold_interval = False

hard_validation_transform = build_transform(True)
hard_validation_dataset = AuroraDataset(config.HARD_VALIDATION_256_DIR, hard_validation_transform)
hard_validation_loader = DataLoader(hard_validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

base_dir = os.path.join(os.path.dirname(__file__))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(base_dir, f"test_results/test_results_{timestamp}.txt")

#testing on hard validation dataset or test dataset
if is_hard_validation_test:
    print("Test of model inference on hard validation dataset")
    data_loader = hard_validation_loader
else:
    print("Test of model inference on test dataset")
    data_loader = test_loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = os.path.join(base_dir, "best_model_pt", "best_model.pt")

model = AuroraCNN()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# =========================
# EVALUATION LOOP
# =========================
criterion = nn.BCEWithLogitsLoss()

all_labels = []
all_preds = []
all_states = []
losses = []

#if decision threshold has intervals for values true/false/uncertain, counting metrics and confusion matrix is different
if is_decision_threshold_interval:
    print("Test with threshold decision intervals started")
    T_FALSE = config.SIGMOID_THRESHOLDS[0]
    T_TRUE = config.SIGMOID_THRESHOLDS[1]

    with (torch.inference_mode()):
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(imgs)

            loss = criterion(outputs, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(outputs).squeeze(1)

            for p, y in zip(probs, labels):
                p = p.item()
                y = y.item()

                all_labels.append(y)

                if p <= T_FALSE:
                    all_preds.append(0)
                    all_states.append("false")

                elif p >= T_TRUE:
                    all_preds.append(1)
                    all_states.append("true")

                else:
                    all_preds.append(None)
                    all_states.append("uncertain")
    #counting confusion matrix
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

    print("===== INTERVAL CONFUSION MATRIX =====")
    print(f"TN: {TN}")
    print(f"FP: {FP}")
    print(f"FN: {FN}")
    print(f"TP: {TP}")
    print(f"Uncertain (False): {U_F}")
    print(f"Uncertain (True):  {U_T}")
    print(f"Total uncertain:   {U_F + U_T}")
    uncertain_rate = (U_F + U_T) / len(all_labels)
    print(f"Uncertain rate: {uncertain_rate:.2%}")

    #counting metrics
    decided_labels = []
    decided_preds = []

    for y, pred in zip(all_labels, all_preds):
        if pred is not None:
            decided_labels.append(y)
            decided_preds.append(pred)

    decided_labels = np.array(decided_labels)
    decided_preds = np.array(decided_preds)

    test_loss = np.mean(losses)
    accuracy = accuracy_score(decided_labels, decided_preds)
    precision = precision_score(decided_labels, decided_preds)
    recall = recall_score(decided_labels, decided_preds)
    f1 = f1_score(decided_labels, decided_preds)
    cm = None

    with open(log_path, "w") as f:
            f.write("===== INTERVAL CONFUSION MATRIX =====\n")
            f.write(f"TN: {TN}\n")
            f.write(f"FP: {FP}\n")
            f.write(f"FN: {FN}\n")
            f.write(f"TP: {TP}\n")
            f.write(f"Uncertain (False): {U_F}\n")
            f.write(f"Uncertain (True):  {U_T}\n")
            f.write(f"Total uncertain:   {U_F + U_T}\n")
            uncertain_rate = (U_F + U_T) / len(all_labels)
            f.write(f"Uncertain rate: {uncertain_rate:.2%}\n")

            f.write("===== TEST RESULTS =====\n")
            f.write(f"Loss:      {test_loss:.4f}\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1 score:  {f1:.4f}\n\n")

    print("===== TEST RESULTS =====")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
#test with binary decision threshold (true/false)
else:
    print("Test started")
    with torch.inference_mode():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            probs = torch.sigmoid(outputs)
            preds = (probs > config.SIGMOID_THRESHOLD_TEST).int()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # counting metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    test_loss = np.mean(losses)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("Test finished")

    print("===== TEST RESULTS =====")
    print(f"Loss:      {test_loss:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    # =========================
    # SAVE RESULTS TO FILE
    # =========================
    os.makedirs("test_results", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("===== TEST RESULTS =====\n")
        f.write(f"Loss:      {test_loss:.4f}\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1 score:  {f1:.4f}\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n")

print(f"Results saved to: {log_path}")