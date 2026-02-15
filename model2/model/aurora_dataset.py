import os
from pathlib import Path

import numpy as np
from PIL import Image

import pandas as pd

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from model2.common.data_preprocessing import PadToSquare, GaussianNoise
from model2 import config

class AuroraDataset(Dataset):

    IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.transform = transform
        csv_path = config.get_labels_csv(self.dataset_dir)
        self.labels = pd.read_csv(csv_path)
        self.labels_dict = {}
        for _, row in self.labels.iterrows():
            file_name = row["filename"]
            label = row[config.CLASS_NAMES].values.astype(np.float32)
            self.labels_dict[file_name] = torch.tensor(label)
        self.image_files = [
            f for f in os.listdir(self.dataset_dir)
            if f in self.labels_dict
               and os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.dataset_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        label = self.labels_dict[img_name]

        if self.transform:
            img = self.transform(img)
        return img, label

#create transformation
def build_transform(test):
    if config.IS_128x128:
        pdaToSquareSize = 128
    else:
        pdaToSquareSize = 256
    if test:
        # transform for test without augmentation
        return transforms.Compose([
            PadToSquare(pdaToSquareSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        # transform with augmentations for training
        transform_components = []
        transform_components.append(PadToSquare(pdaToSquareSize))
        if config.ROTATION_APPLY:
            transform_components.append(transforms.RandomApply([transforms.RandomRotation(degrees=5,
                                                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                                                          expand=False, fill=0)], p=config.ROTATION_P))
        if config.COLOR_JITTER_APPLY:
            transform_components.append(transforms.RandomApply([transforms.ColorJitter(brightness=config.BRIGHTNESS, contrast=config.CONTRAST)],
                                                               p=config.COLOR_JITTER_P))
        transform_components.append(transforms.ToTensor())
        if config.GAUSSIAN_NOISE_APPLY:
            transform_components.append(transforms.RandomApply([GaussianNoise(config.GAUSSIAN_NOISE)],
                                                               p = config.GAUSSIAN_NOISE_P))
        transform_components.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        return transforms.Compose(transform_components)

#transformations
train_transform = build_transform(False)
eval_transform = build_transform(True)

#dataloader
if config.IS_128x128:
    train_dataset = AuroraDataset(config.TRAIN_128_DIR, train_transform)
    validation_dataset = AuroraDataset(config.VAL_128_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
else:
    train_dataset = AuroraDataset(config.TRAIN_256_DIR, train_transform)
    validation_dataset = AuroraDataset(config.VAL_256_DIR, eval_transform)
    test_dataset = AuroraDataset(config.TEST_256_DIR, eval_transform)
    hard_validation_dataset = AuroraDataset(config.HARD_VALIDATION_256_DIR, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    hard_validation_loader = DataLoader(hard_validation_dataset, batch_size=config.BATCH_SIZE, shuffle=False)



