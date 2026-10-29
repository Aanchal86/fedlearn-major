
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset, random_split

from federated.data.dataset_loader import load_dataset


@dataclass
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset


def get_dataset_splits(
    dataset_name: str,
    seed: int = 42,
) -> DatasetSplits:
    """
    Return reproducible training, validation,
    and testing datasets.

    Fashion-MNIST:
        54,000 train
         6,000 validation
        10,000 test

    PneumoniaMNIST:
        Uses official train/val/test splits.
    """

    if dataset_name == "fashion_mnist":
        full_train = load_dataset(
            dataset_name,
            split="train",
        )

        test_dataset = load_dataset(
            dataset_name,
            split="test",
        )

        train_size = int(0.9 * len(full_train))
        val_size = len(full_train) - train_size

        generator = torch.Generator().manual_seed(seed)

        train_dataset, val_dataset = random_split(
            full_train,
            [train_size, val_size],
            generator=generator,
        )

        return DatasetSplits(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset,
        )

    if dataset_name == "pneumonia_mnist":
        return DatasetSplits(
            train=load_dataset(dataset_name, "train"),
            val=load_dataset(dataset_name, "val"),
            test=load_dataset(dataset_name, "test"),
        )

    raise ValueError(
        f"Unsupported dataset: {dataset_name}"
    )
