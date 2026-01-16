import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from model1.model.aurora_dataset import train_loader, validation_loader
from model1.model.cnn_model import AuroraCNN


#train, eval functions
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses, preds, targets = [], [], []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.append(torch.sigmoid(logits).detach().cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    bin_preds = (preds>0.5).int()

    return {
        "loss": np.mean(losses),
        "accuracy": (bin_preds == targets).float().mean().item(),
        "f1": f1_score(targets, bin_preds),
        "precision": precision_score(targets, bin_preds),
        "recall": recall_score(targets, bin_preds),
    }

#validation/test
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds.append(torch.sigmoid(logits).cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    bin_preds = (preds>0.5).int()

    return {
        "loss": np.mean(losses),
        "accuracy": (bin_preds == targets).float().mean().item(),
        "f1": f1_score(targets, bin_preds),
        "precision": precision_score(targets, bin_preds),
        "recall": recall_score(targets, bin_preds)
    }

#training process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AuroraCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=0.0003, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=6, min_lr=0.000001)
num_epochs = 100
patience = 12
best_f1 = 0.0
patience_counter = 0

result = {
    "train_loss":[], "val_loss":[],
    "train_accuracy":[], "val_accuracy":[],
    "train_f1":[], "val_f1":[],
    "train_precision":[], "val_precision":[],
    "train_recall":[], "val_recall":[],
    "lr": []
}

for epoch in range(num_epochs):
    train_m = train_epoch(model, train_loader, optimizer, criterion, device)
    val_m = eval_epoch(model, validation_loader, optimizer, criterion, device)
    scheduler.step(val_m["loss"])

    for k in ["loss", "accuracy", "f1", "precision", "recall"]:
        result[f"train_{k}"].append(train_m[k])
        result[f"val_{k}"].append(val_m[k])

    lr = optimizer.param_groups[0]["lr"]
    result[f"lr"].append(lr)

    print(
        f"Epoch {epoch+1:03d} | "
        f"LR {lr: .2e} |"
        f"Train F1 {train_m['f1']:.3f} |"
        f"Val F1 {val_m['f1']:.3f}"
    )

    if val_m["f1"] > best_f1:
        best_f1 = val_m["f1"]
        patience_counter = 0
        torch.save(model.state_dict(), f"model1.{epoch+1:03d}.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1:03d}.")
            break

# plots
print(f"Train accuracy: {result["train_accuracy"][-1]}, "
      f"Val accuracy: {result["val_accuracy"][-1]}")

plt.figure(figsize =(12,8))
plt.plot(result["train_loss"], label="train loss")
plt.plot(result["val_loss"], label="val loss")
plt.legend()
plt.show()

plt.figure(figsize =(12,8))
plt.plot(result["train_accuracy"], label="train acc")
plt.plot(result["val_accuracy"], label="val acc")
plt.legend()
plt.show()

plt.figure(figsize =(12,8))
plt.plot(result["val_f1"], label="val f1")
plt.plot(result["val_precision"], label="val precision")
plt.plot(result["val_recall"], label="val recall")
plt.legend()
plt.show()
