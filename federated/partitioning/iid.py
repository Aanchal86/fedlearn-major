
from collections import Counter

import numpy as np
from torch.utils.data import Dataset, Subset


def partition_iid(
    dataset: Dataset,
    num_clients: int = 5,
    seed: int = 42,
) -> dict[int, Subset]:

    if num_clients < 1:
        raise ValueError("num_clients must be positive")

    if len(dataset) < num_clients:
        raise ValueError(
            "Dataset must contain at least one sample per client"
        )

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(dataset))
    partitions = np.array_split(shuffled_indices, num_clients)

    return {
        client_id: Subset(dataset, indices.tolist())
        for client_id, indices in enumerate(partitions)
    }


def get_label_distribution(dataset: Dataset) -> dict[int, int]:
    counts = Counter()

    for index in range(len(dataset)):
        _, label = dataset[index]
        label_value = int(np.asarray(label).item())
        counts[label_value] += 1

    return dict(sorted(counts.items()))
