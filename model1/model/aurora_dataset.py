import os
from PIL import Image

import pandas as pd

import torch
from mpmath.identification import transforms
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import contrast

from model1.model.data_preprocessing import PadToSquare, GaussianNoise


class AuroraDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        labels_suffix= self.root_dir.split('_')[0]
        csv_path = os.path.join(root_dir,f"labels_{labels_suffix}.csv")
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row["filename"])
        label = torch.tensor(row["label"], dtype=torch.float32)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

#transformations
train_transform = transforms.Compose([PadToSquare(256),
                                      transforms.RandomApply([transforms.RandomRotation(8)], p = 0.4),
                                      transforms.RandomApply([transforms.ColorJitter(brightness = 0.08, contrast = 0.08)], p = 0.7),
                                      transforms.RandomApply([GaussianNoise(0.01)], p = 0.3),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])

eval_transform = transforms.Compose([PadToSquare(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                     ])

#dataloader
train_dataset = AuroraDataset("train_256", train_transform)
validation_dataset = AuroraDataset("validation_256", eval_transform)
test_dataset = AuroraDataset("test_256", eval_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
