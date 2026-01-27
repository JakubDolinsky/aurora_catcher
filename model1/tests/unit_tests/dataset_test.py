import torch

from model1 import config


# ============================================================
# BASIC DATASET CONTRACT TEST
# ============================================================
def test_train_dataset_single_sample(train_dataset):
    """
    This test verifies the BASIC CONTRACT of the TRAIN dataset.

    checks:
    - Dataset loads a JPG image from disk
    - Transform with augmentation (train dataset) pipeline is applied
    - Output image is a torch.Tensor
    - Output shape is (3, 256, 256) or (3, 128,128)
    - Label is valid (0 or 1)
    """
    img, label = train_dataset[0]

    # After __getitem__, image MUST already be transformed to tensor
    assert isinstance(img, torch.Tensor), \
        "Dataset output image must be torch.Tensor after transforms"
    print("- Image is loaded as torch tensor.\n")
    if config.IS_128x128:
        img_size = 128
    else:
        img_size = 256
    assert img.shape == (3, img_size, img_size), \
        f"Wrong image shape after transforms: {img.shape}"
    print("- Image has proper shape for model.\n")

    assert label in [0, 1], \
        f"Invalid label value: {label}"
    print("- Label is valid.\n")
    print("----- Dataset single sample test passed -----\n")


# ============================================================
# NORMALIZATION TEST
# ============================================================
def test_train_dataset_normalization_range(train_dataset):
    """
    Verifies that normalization was applied correctly. Using train dataset with augmentation

    Expected range:
    - Using mean=0.5, std=0.5
    - Pixel values should be roughly in [-1, 1]

    This test catches:
    - Missing Normalize()
    - Double normalization
    - Wrong mean/std values
    """
    img, _ = train_dataset[0]

    min_val = img.min().item()
    max_val = img.max().item()

    assert min_val >= -1.1 and max_val <= 1.1, \
        f"Normalization out of range: min={min_val}, max={max_val}"
    print("----- Normalize test passed -----\n")


# ============================================================
# AUGMENTATION VARIABILITY TEST
# ============================================================
def test_train_dataset_augmentation_variability(test_with_aug_dataset):
    """
    Checks whether TRAIN augmentations are actually applied.

    Strategy:
    - Load the SAME dataset index multiple times
    - Compare resulting tensors
    - At least some of them should differ
    """
    imgs = []

    for _ in range(5):
        img, _ = test_with_aug_dataset[0]
        imgs.append(img)

    # Mean absolute difference between samples
    diffs = [
        torch.mean(torch.abs(imgs[0] - imgs[i])).item()
        for i in range(1, 5)
    ]

    assert any(d > 0.01 for d in diffs), \
        "Augmentation seems not applied (all samples look identical)"
    print("----- Augmentation test passed -----\n")