import torch

from model2 import config


def test_model_forward(model):
    if config.IS_128x128:
        img_size = 128
    else:
        img_size = 256
    x = torch.randn(2, 3, img_size, img_size)
    y = model(x)

    assert y.shape == (2, len(config.CLASS_NAMES)), f"Wrong output shape: {y.shape}"
    assert not torch.isnan(y).any(), "NaN in model output"
    print("----- Test of forward pass passed. -----\n")

def test_gap_removes_spatial_dims(model):
    if config.IS_128x128:
        img_size = 128
    else:
        img_size = 256
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        features = model.features(x)
        gap = model.gap(features)
        logits = model.classifier(gap)

    assert gap.shape == (1, features.shape[1], 1, 1), f"GAP output shape wrong: {gap.shape}"
    assert logits.shape[1] == len(config.CLASS_NAMES), f"Classifier output wrong: {logits.shape}"
    print("----- Test of GAP passed. ------\n")

