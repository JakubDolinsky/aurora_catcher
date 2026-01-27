from pathlib import Path

import shutil

import pandas as pd
from tqdm import tqdm
import csv
import datetime

import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn

from model1 import config
from model1.model.aurora_dataset import train_loader, validation_loader
from model1.model.cnn_model import AuroraCNN


# -------------------- TRAIN / EVAL --------------------

def train_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    losses, preds, targets = [], [], []

    pbar = tqdm(
        loader,
        desc=f"Train epoch [{epoch + 1}/{num_epochs}]",
        leave=True,
        dynamic_ncols=True
    )

    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.append(torch.sigmoid(logits).detach().cpu())
        targets.append(y.cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    bin_preds = (preds > 0.5).int()

    return {
        "loss": np.mean(losses),
        "accuracy": (bin_preds == targets).float().mean().item(),
        "f1": f1_score(targets, bin_preds),
        "precision": precision_score(targets, bin_preds),
        "recall": recall_score(targets, bin_preds),
    }


# validation/test
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds.append(torch.sigmoid(logits).cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    bin_preds = (preds > 0.5).int()

    return {
        "loss": np.mean(losses),
        "accuracy": (bin_preds == targets).float().mean().item(),
        "f1": f1_score(targets, bin_preds),
        "precision": precision_score(targets, bin_preds),
        "recall": recall_score(targets, bin_preds)
    }


def log(msg, log_file):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


def plot_metrics(metrics_path, save_dir):
    df = pd.read_csv(metrics_path)

    # 1️. Train/Val Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'loss_plot.png')
    plt.close()

    # 2️. Train/Val Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train/Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'accuracy_plot.png')
    plt.close()

    # 3️. Train/Val F1, Precision, Recall
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_f1'], label='Train F1')
    plt.plot(df['epoch'], df['val_f1'], label='Val F1')
    plt.plot(df['epoch'], df['train_precision'], label='Train Precision', linestyle='--')
    plt.plot(df['epoch'], df['val_precision'], label='Val Precision', linestyle='--')
    plt.plot(df['epoch'], df['train_recall'], label='Train Recall', linestyle=':')
    plt.plot(df['epoch'], df['val_recall'], label='Val Recall', linestyle=':')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Train/Validation F1, Precision, Recall')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'f1_precision_recall_plot.png')
    plt.close()


# -------------------- MAIN --------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("checkpoints") / f"run_{run_id}"
    best_models_dir = run_dir / "best_models"
    final_dir = run_dir / "final"

    best_models_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    log_file = final_dir / "training.log"
    metrics_path = final_dir / "metrics.csv"

    # ---- HEADER LOG ----
    log("=" * 80, log_file)
    log(f"Training started at {run_id}", log_file)
    log(f"Device: {device}", log_file)
    log(f"Epochs: {config.EPOCHS}", log_file)
    log(f"Batch size: {config.BATCH_SIZE}", log_file)
    log(f"LR: {config.LR}, Weight decay: {config.WEIGHT_DECAY}", log_file)
    log("=" * 80, log_file)

    model = AuroraCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=1e-6
    )
    num_epochs = config.EPOCHS
    patience = config.PATIENCE
    best_f1 = 0.0
    patience_counter = 0

    # ---- CSV INIT ----
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_acc", "train_f1", "train_precision", "train_recall",
            "val_loss", "val_acc", "val_f1", "val_precision", "val_recall",
            "lr"
        ])

    for epoch in range(num_epochs):


        train_m = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
        val_m = eval_epoch(model, validation_loader, criterion, device)

        scheduler.step(val_m["loss"])
        lr = optimizer.param_groups[0]["lr"]

        # ---- METRICS LOG ----
        log(f"\n--- Epoch {epoch + 1}/{num_epochs} ---", log_file)
        log(
            f"Train | loss={train_m['loss']:.4f}, acc={train_m['accuracy']:.3f}, "
            f"F1={train_m['f1']:.3f}, P={train_m['precision']:.3f}, R={train_m['recall']:.3f}",
            log_file
        )

        log(
            f"Val   | loss={val_m['loss']:.4f}, acc={val_m['accuracy']:.3f}, "
            f"F1={val_m['f1']:.3f}, P={val_m['precision']:.3f}, R={val_m['recall']:.3f}",
            log_file
        )

        # ---- CSV WRITE ----
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_m["loss"], train_m["accuracy"], train_m["f1"],
                train_m["precision"], train_m["recall"],
                val_m["loss"], val_m["accuracy"], val_m["f1"],
                val_m["precision"], val_m["recall"],
                lr
            ])

        # ---- CHECKPOINTING ----
        if val_m["f1"] > best_f1:
            log(
                f"Validation F1 improved {best_f1:.4f} -> {val_m['f1']:.4f}. Saving model.",
                log_file
            )
            best_f1 = val_m["f1"]
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_models_dir / f"epoch_{epoch + 1:03d}_f1_{best_f1:.3f}.pt"
            )
            torch.save(model.state_dict(), final_dir / "best_model.pt")
        else:
            patience_counter += 1
            log(
                f"No improvement in F1. Patience {patience_counter}/{patience}.",
                log_file
            )

            if patience_counter >= patience:
                log("Early stopping triggered.", log_file)
                break

        # ---- SAVE CONFIG FOR REPRODUCIBILITY ----
        config_dst = final_dir / "config.py"
        shutil.copy(config.__file__, config_dst)

    # ---- PLOTTING METRICS ----
    log("\nGenerating and saving metric plots...", log_file)
    plot_metrics(metrics_path, final_dir)
    log("Plots saved in final directory.", log_file)

    log("Training finished.", log_file)


if __name__ == "__main__":
    main()
