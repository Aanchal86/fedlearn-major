
import torch
from torch import nn

from federated.data.preprocessing import get_dataset_config


class SmallCNN(nn.Module):
    """
    CNN for 28x28 grayscale image classification.

    Supports Fashion-MNIST and PneumoniaMNIST.
    """

    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        if input_channels < 1:
            raise ValueError("input_channels must be positive")

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_model(dataset_name: str) -> SmallCNN:
    """
    Create the correct CNN for a supported dataset.
    """

    config = get_dataset_config(dataset_name)

    return SmallCNN(
        num_classes=config["num_classes"],
        input_channels=config["input_channels"],
    )
