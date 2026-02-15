import matplotlib.pyplot as plt

from model2 import config
from model2.model.aurora_dataset import AuroraDataset, eval_transform


def debug_visualize_train_sample(train_dataset, idx=0):
    """
    Manual visual sanity check.

    What to visually confirm:
    - Image is square (256x256 or 128x128)
    - Padding is correct (no stretching)
    - Augmentation sometimes visible
    - Colors look realistic (after denormalization)

    This is NOT an automated test.
    """

    img, label = train_dataset[idx]

    # Convert CHW → HWC
    img = img.permute(1, 2, 0)

    # Denormalize (mean=0.5, std=0.5)
    img = img * 0.5 + 0.5
    img = img.clamp(0, 1)

    plt.imshow(img.numpy())
    plt.title(f"Train sample idx={idx}, label={label}")
    plt.axis("off")
    plt.show()

def main():
    if config.IS_128x128:
        dataset = AuroraDataset(config.TRAIN_128_DIR, eval_transform)
    else:
        dataset = AuroraDataset(config.TRAIN_256_DIR, eval_transform)

    debug_visualize_train_sample(dataset, 3236)

if __name__ == "__main__":
    main()