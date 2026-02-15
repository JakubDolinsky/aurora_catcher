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

from model2 import config
from model2.model.aurora_dataset import train_loader, validation_loader
from model2.model.cnn_model import AuroraCNN


# -------------------- TRAIN / EVAL --------------------

def train_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs, thresholds):
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
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.append(torch.sigmoid(logits).detach().cpu())
        targets.append(y.cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    bin_preds = (preds > thresholds).int()
    bin_preds_train = (preds > 0.5).int()
    mean_probs = preds.mean(dim=0).numpy()


    return {
        "loss": np.mean(losses),
        "accuracy": (bin_preds == targets).float().mean().item(),
        #f1 calculated for metrics reporting purposes -relevant values
        "f1": f1_score(targets, bin_preds, average="samples", zero_division=0),
        #f1 calculated for early stopping purposes, can not be dependent on threshold business settings
        "f1_train": f1_score(targets, bin_preds_train, average="samples", zero_division=0),
        "precision": precision_score(targets, bin_preds, average="samples", zero_division=0),
        "recall": recall_score(targets, bin_preds, average="samples", zero_division=0),
        "mean_probs": mean_probs
    }


# validation/test
@torch.no_grad()
def eval_epoch(model, loader, criterion, device, thresholds):
    model.eval()
    losses, preds, targets = [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())
        preds.append(torch.sigmoid(logits).cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    bin_preds = (preds > thresholds).int()
    bin_preds_train = (preds>0.5).int()
    mean_probs = preds.mean(dim=0).numpy()

    return {
        "loss": np.mean(losses),
        "accuracy": (bin_preds == targets).float().mean().item(),
        "f1": f1_score(targets, bin_preds, average="samples",zero_division=0),
        "f1_train": f1_score(targets, bin_preds_train, average="samples", zero_division=0),
        "precision": precision_score(targets, bin_preds, average="samples",zero_division=0),
        "recall": recall_score(targets, bin_preds, average="samples",zero_division=0),
        "mean_probs": mean_probs
    }

def compute_class_prevalence_from_dataset(dataset, num_classes):
    counts = torch.zeros(num_classes)
    total = 0
    for _, labels in dataset:
        counts += labels.float()
        total += 1
    return (counts / total).numpy()

def log(msg, log_file):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


def plot_metrics(metrics_path, save_dir,real_train_prevalence, real_val_prevalence):
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

    #4. plot class probability trends
    epochs = df["epoch"]

    for i, cls in enumerate(config.CLASS_NAMES):
        plt.figure(figsize=(10, 5))

        plt.plot(epochs, df[f"train_prob_{cls}"], label="Train mean prob")
        plt.plot(epochs, df[f"val_prob_{cls}"], label="Val mean prob")

        plt.axhline(
            y=real_train_prevalence[i],
            linestyle="--",
            label="Train prevalence"
        )
        plt.axhline(
            y=real_val_prevalence[i],
            linestyle=":",
            label="Val prevalence"
        )

        plt.title(f"Mean predicted probability – {cls}")
        plt.xlabel("Epoch")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()

        plt.savefig(save_dir / f"prob_trend_{cls}.png")
        plt.close()

# -------------------- MAIN --------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_checkpoint_dir = Path("checkpoints")
    root_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_checkpoint_dir / f"run_{run_id}"
    best_models_dir = run_dir / "best_models"
    final_dir = run_dir / "final"

    best_models_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    log_file = final_dir / "training.log"
    metrics_path = final_dir / "metrics.csv"

    #rate of positive cases of categories in dataset
    real_train_prevalence = compute_class_prevalence_from_dataset(
        train_loader.dataset,
        num_classes=len(config.CLASS_NAMES)
    )

    real_val_prevalence = compute_class_prevalence_from_dataset(
        validation_loader.dataset,
        num_classes=len(config.CLASS_NAMES)
    )

    thresholds = torch.tensor(config.SIGMOID_THRESHOLDS, dtype=torch.float32)
    thresholds = thresholds.to(device)
    # ---- HEADER LOG ----
    log("=" * 80, log_file)
    log(f"Training started at {run_id}", log_file)
    log(f"Device: {device}", log_file)
    log(f"Epochs: {config.EPOCHS}", log_file)
    log(f"Batch size: {config.BATCH_SIZE}", log_file)
    log(f"LR: {config.LR}, Weight decay: {config.WEIGHT_DECAY}", log_file)
    log(f"Real train prevalence: {real_train_prevalence}, Real validation prevalence: {real_val_prevalence}", log_file)
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
            *[f"train_prob_{c}" for c in config.CLASS_NAMES],
            "val_loss", "val_acc", "val_f1", "val_precision", "val_recall",
            *[f"val_prob_{c}" for c in config.CLASS_NAMES],
            "lr"
        ])

    for epoch in range(num_epochs):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs,thresholds)
        val_m = eval_epoch(model, validation_loader, criterion, device,thresholds)

        scheduler.step(val_m["loss"])
        lr = optimizer.param_groups[0]["lr"]

        train_mean_probs_str = ", ".join(
            f"{cls}:{p:.3f}"
            for cls, p in zip(config.CLASS_NAMES, train_m["mean_probs"]))
        val_mean_probs_str = ", ".join(
            f"{cls}:{p:.3f}"
            for cls, p in zip(config.CLASS_NAMES, val_m["mean_probs"]))
        # ---- METRICS LOG ----
        log(f"\n--- Epoch {epoch + 1}/{num_epochs} ---", log_file)
        log(
            f"Train | loss={train_m['loss']:.4f}, acc={train_m['accuracy']:.3f}, "
            f"F1={train_m['f1']:.3f}, P={train_m['precision']:.3f}, R={train_m['recall']:.3f}, M_PROBS = {train_mean_probs_str}",
            log_file
        )

        log(
            f"Val   | loss={val_m['loss']:.4f}, acc={val_m['accuracy']:.3f}, "
            f"F1={val_m['f1']:.3f}, P={val_m['precision']:.3f}, R={val_m['recall']:.3f}, M_PROBS = {val_mean_probs_str}",
            log_file
        )

        # ---- CSV WRITE ----
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_m["loss"], train_m["accuracy"], train_m["f1"],
                train_m["precision"], train_m["recall"],*train_m["mean_probs"],
                val_m["loss"], val_m["accuracy"], val_m["f1"],
                val_m["precision"], val_m["recall"],*val_m["mean_probs"],
                lr
            ])

        # ---- CHECKPOINTING ----
        if val_m["f1_train"] > best_f1:
            log(
                f"Validation F1 improved {best_f1:.4f} -> {val_m['f1_train']:.4f}. Saving model.",
                log_file
            )
            best_f1 = val_m["f1_train"]
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
    plot_metrics(metrics_path, final_dir,real_train_prevalence, real_val_prevalence)
    log("Plots saved in final directory.", log_file)

    log("Training finished.", log_file)


if __name__ == "__main__":
    main()
