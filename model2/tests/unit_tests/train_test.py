import torch
import torch.nn as nn

def test_full_pipeline(dataloader, model):
    criterion = nn.BCEWithLogitsLoss()

    imgs, labels = next(iter(dataloader))
    labels = labels.float()

    outputs = model(imgs)
    loss = criterion(outputs, labels)

    assert loss.item() > 0
    assert not torch.isnan(loss), "Loss is NaN"
    print("----- Full pipeline test passed -----\n")

