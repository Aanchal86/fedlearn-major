
from pathlib import Path

import medmnist
from medmnist import INFO
from torchvision.datasets import FashionMNIST

from federated.data.preprocessing import get_transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_dataset(
    dataset_name: str,
    split: str = "train",
    download: bool = True,
):
    """
    Load a supported dataset using a common interface.

    Supported splits:
        Fashion-MNIST: train, test
        PneumoniaMNIST: train, val, test
    """
    transform = get_transforms(dataset_name)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if dataset_name == "fashion_mnist":
        if split not in ("train", "test"):
            raise ValueError(
                "Fashion-MNIST supports train and test only"
            )

        return FashionMNIST(
            root=str(DATA_DIR),
            train=(split == "train"),
            transform=transform,
            download=download,
        )

    if dataset_name == "pneumonia_mnist":
        if split not in ("train", "val", "test"):
            raise ValueError(
                "Invalid PneumoniaMNIST split"
            )

        info = INFO["pneumoniamnist"]
        dataset_class = getattr(
            medmnist,
            info["python_class"]
        )

        return dataset_class(
            split=split,
            root=str(DATA_DIR),
            transform=transform,
            download=download,
            size=28,
        )

    raise ValueError(
        f"Unsupported dataset: {dataset_name}"
    )
