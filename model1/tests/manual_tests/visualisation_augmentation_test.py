import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model1 import config
from model1.model.aurora_dataset import PadToSquare
from pathlib import Path

from model1.model.data_preprocessing import GaussianNoise


# ------------------------------------------------------------------
# PROPOSED AUGMENTATION
# ------------------------------------------------------------------
def proposed_train_transform(pad_size):
    transform_components = []

    transform_components.append(PadToSquare(pad_size))
    if config.ROTATION_APPLY:
        transform_components.append(transforms.RandomApply([transforms.RandomRotation(degrees=5,
    interpolation=transforms.InterpolationMode.BICUBIC, expand=False, fill=0)], p=config.ROTATION_P))
    if config.COLOR_JITTER_APPLY:
        transform_components.append(
            transforms.RandomApply([transforms.ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST)],
                                   p=config.COLOR_JITTER_P))
    transform_components.append(transforms.ToTensor())
    if config.GAUSSIAN_NOISE_APPLY:
        transform_components.append(transforms.RandomApply([GaussianNoise(config.GAUSSIAN_NOISE)], p=config.GAUSSIAN_NOISE_P))
    transform_components.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(transform_components)

# ------------------------------------------------------------------
# NO AUGMENTATION
# ------------------------------------------------------------------
def no_augmentation_train_transform(pad_size):
    return transforms.Compose([
        PadToSquare(pad_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def denormalize(img):
    return (img * 0.5 + 0.5).clamp(0, 1)

def load_pil_image(idx=0):
    root = config.TRAIN_128_DIR if config.IS_128x128 else config.TRAIN_256_DIR
    image_paths = sorted(Path(root).rglob("*.jpg"))
    return Image.open(image_paths[idx]).convert("RGB")

# ------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------
def visualize_sample(idx=0, n_aug=5):
    pad_size = 128 if config.IS_128x128 else 256

    img_pil = load_pil_image(idx)

    fig, axes = plt.subplots(2, n_aug, figsize=(4 * n_aug, 10))

    # --------------------------------------------------------------
    # NO AUG
    # --------------------------------------------------------------
    no_aug_tf = no_augmentation_train_transform(pad_size)

    for j in range(n_aug):
        img = no_aug_tf(img_pil)
        img = denormalize(img).permute(1, 2, 0)
        axes[0, j].imshow(img)
        axes[0, j].axis("off")
        if j == 0:
            axes[0, j].set_title("NO AUG")

    # --------------------------------------------------------------
    # PROPOSED AUG
    # --------------------------------------------------------------
    proposed_tf = proposed_train_transform(pad_size)
    for j in range(n_aug):
        img = proposed_tf(img_pil)
        img = denormalize(img).permute(1, 2, 0)
        axes[1, j].imshow(img)
        axes[1, j].axis("off")
        if j == 0:
            axes[1, j].set_title("PROPOSED AUG")

    plt.suptitle(f"Augmentation comparison | idx={idx}", fontsize=14)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    visualize_sample(idx=0, n_aug=5)

if __name__ == "__main__":
    main()
