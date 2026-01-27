import torch
from torch.utils.data import DataLoader

from model1 import config
from model1.model.aurora_dataset import AuroraDataset, eval_transform
from model1.model.cnn_model import AuroraCNN


def debug_overfit_small_batch(model, dataset):
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = torch.nn.BCEWithLogitsLoss()

    samples = [dataset[i] for i in range(5)]
    imgs = torch.stack([s[0] for s in samples])
    labels = torch.tensor([s[1] for s in samples]).float().unsqueeze(1)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

    assert loss.item() < 0.05, \
        "Model failed to overfit 5 samples → architecture or data bug"
    print("Overfit test passed")

def main():
    if config.IS_128x128:
        dataset = AuroraDataset(config.TRAIN_128_DIR, eval_transform)
    else:
        dataset = AuroraDataset(config.TRAIN_256_DIR, eval_transform)

    model = AuroraCNN()
    debug_overfit_small_batch(model, dataset)

if __name__ == "__main__":
    main()

