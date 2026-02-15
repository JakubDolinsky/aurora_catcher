from pathlib import Path

import pandas as pd
from tqdm import tqdm
import csv
import datetime

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

from model2 import config
from model2.model.aurora_dataset import train_loader
from model2.model.cnn_model import AuroraCNN

# mean count of positive labels in validation dataset is used for calculating alpha
#calculation by formula: alpha = 0.1* (1-mean count/count of classes)
ALPHA =0.18

def train_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs, final_dir, log_file):
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
        labels = y.float()

        loss_matrix = criterion(logits, torch.where(labels == -1, torch.zeros_like(labels), labels.float()))
        pos_mask = (labels == 1)
        unl_mask = (labels == -1)
        loss_pos = (loss_matrix * pos_mask.float()).sum() / pos_mask.sum()
        loss_unl = (loss_matrix * unl_mask.float()).sum() / unl_mask.sum()

        loss = loss_pos + ALPHA * loss_unl
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.append(torch.sigmoid(logits).detach().cpu())
        targets.append(y.cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    bin_preds = (preds > 0.5).int()
    mask = targets != -1
    masked_bin_preds = bin_preds[mask]
    masked_targets = targets[mask]
    acc = (masked_bin_preds == masked_targets).float().mean().item()

    epoch_confidence = preds
    mean_conf_per_class = epoch_confidence.mean(dim=0).tolist()
    avg_pred_pos_per_sample = (bin_preds.sum(dim=1).float().mean()).item()

    # Print & log
    msg = (
        f"Epoch {epoch + 1}/{num_epochs}:\n"
        f"Train loss: {np.mean(losses):.4f}\n"
        f"Loss_pos: {loss_pos.item():.4f}, Loss_unl: {loss_unl.item():.4f}\n"
        f"Mean confidence per class: {mean_conf_per_class}\n"
        f"Avg predicted positives per sample: {avg_pred_pos_per_sample:.2f}\n"
    )
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")

    # Histogram of sigmoids
    if int(epoch) == num_epochs - 1:
        plt.figure(figsize=(8, 4))
        plt.hist(epoch_confidence.flatten().numpy(), bins=20, range=(0, 1))
        plt.title(f"Confidence histogram - epoch {epoch + 1}")
        plt.xlabel("Sigmoid output")
        plt.ylabel("Count")
        plt.savefig(final_dir / f'confidence_histogram_{epoch + 1}.png')
        plt.close()

    return {
        "loss": np.mean(losses),
        "accuracy": acc,
        "loss_pos": loss_pos.item(),
        "loss_unl": loss_unl.item(),
        "mean_conf_per_class": mean_conf_per_class,
        "avg_pred_pos_per_sample": avg_pred_pos_per_sample
    }

def log(msg, log_file):
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")


def plot_metrics(metrics_path, save_dir):
    df = pd.read_csv(metrics_path)

    # 1️. Train Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'loss_plot.png')
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

    # ---- HEADER LOG ----
    log("=" * 80, log_file)
    log(f"Training started at {run_id}", log_file)
    log(f"Device: {device}", log_file)
    log(f"Epochs: {config.EPOCHS}", log_file)
    log(f"LR: {config.LR}, Weight decay: {config.WEIGHT_DECAY}", log_file)
    log("=" * 80, log_file)

    model = AuroraCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
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

    # ---- CSV INIT ----
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_acc",
            "loss_pos", "loss_unl",
            "avg_pred_pos_per_sample"
        ])

    for epoch in range(num_epochs):
        # train epoch
        train_m = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs, final_dir, log_file
        )
        scheduler.step(train_m["loss"])

        # ---- LOG into console and file ----
        log(f"\n--- Epoch {epoch + 1}/{num_epochs} ---", log_file)
        msg = (
            f"Train | loss={train_m['loss']:.4f}, acc={train_m['accuracy']:.3f}, "
            f"loss_pos={train_m['loss_pos']:.4f}, loss_unl={train_m['loss_unl']:.4f}, "
            f"avg_pred_pos_per_sample={train_m['avg_pred_pos_per_sample']:.2f}\n"
            f"Mean confidence per class: {train_m['mean_conf_per_class']}"
        )
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        # ---- CSV zapis ----
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_m["loss"], train_m["accuracy"],
                train_m["loss_pos"], train_m["loss_unl"],
                train_m["avg_pred_pos_per_sample"]
            ])

    # ---- CHECKPOINTING ----
    torch.save(model.state_dict(), final_dir / "best_model.pt")

    # ---- PLOTTING METRICS ----
    log("\nGenerating and saving metric plots...", log_file)
    plot_metrics(metrics_path, final_dir)
    log("Plots saved in final directory.", log_file)
    log("Training finished.", log_file)


if __name__ == "__main__":
    main()
