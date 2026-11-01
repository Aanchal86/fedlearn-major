
import numpy as np
import torch
from torch.utils.data import Dataset


class NoisyLabelDataset(Dataset):
    """
    Dataset wrapper that deterministically corrupts
    a fraction of labels without changing the original data.

    Only use this wrapper for client training datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
        noise_rate: float = 0.3,
        num_classes: int = 10,
        seed: int = 42,
    ):
        if not 0 <= noise_rate <= 1:
            raise ValueError(
                "noise_rate must be between 0 and 1"
            )

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        self.dataset = dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes

        rng = np.random.default_rng(seed)

        num_samples = len(dataset)
        num_corrupted = int(
            round(num_samples * noise_rate)
        )

        corrupted_indices = rng.choice(
            num_samples,
            size=num_corrupted,
            replace=False,
        )

        # Use an independent random offset to ensure
        # the replacement label differs from the original.
        self.corrupted_labels = {}

        for index in corrupted_indices:
            index = int(index)

            _, original_label = dataset[index]
            original_label = int(
                np.asarray(original_label).item()
            )

            if not 0 <= original_label < num_classes:
                raise ValueError(
                    "Dataset label outside class range"
                )

            offset = int(
                rng.integers(1, num_classes)
            )

            new_label = (
                original_label + offset
            ) % num_classes

            self.corrupted_labels[index] = new_label

    def __len__(self):
        return len(self.dataset)

    
    def __getitem__(self, index):
        image, original_label = self.dataset[index]

        if index in self.corrupted_labels:
            new_label = self.corrupted_labels[index]

            if isinstance(original_label, torch.Tensor):
                new_label = torch.as_tensor(
                    new_label,
                    dtype=original_label.dtype,
                    device=original_label.device,
                )
            elif isinstance(original_label, np.ndarray):
                new_label = np.asarray(
                    new_label,
                    dtype=original_label.dtype,
                )
            else:
                new_label = type(original_label)(new_label)

            return image, new_label

        return image, original_label


    @property
    def num_corrupted(self):
        return len(self.corrupted_labels)
