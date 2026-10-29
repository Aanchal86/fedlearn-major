
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

from federated.data.splits import get_dataset_splits


class StandardizedDataset(Dataset):
    """
    Wrap a dataset to standardize labels.

    Output:
        image: torch.Tensor
        label: torch.long scalar
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        # Fashion-MNIST returns an integer label.
        # MedMNIST returns a NumPy array.
        label = np.asarray(label)

        if label.size != 1:
            raise ValueError(
                "Expected a single classification label"
            )

        label = int(label.item())

        return image, torch.tensor(
            label, dtype=torch.long
        )


def create_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    seed: int = 42,
):
    """
    Create train, validation, and test DataLoaders.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    splits = get_dataset_splits(
        dataset_name=dataset_name,
        seed=seed,
    )

    train_dataset = StandardizedDataset(splits.train)
    val_dataset = StandardizedDataset(splits.val)
    test_dataset = StandardizedDataset(splits.test)

    # Separate generator for reproducible shuffling.
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader
