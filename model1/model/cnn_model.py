import torch.nn as nn
from torch.nn import BatchNorm2d


class AuroraCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,3,padding=1),
            BatchNorm2d(64),
            nn.Dropout2d(0.15),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,padding=1),
            BatchNorm2d(128),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,padding=1),
            BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256,1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x