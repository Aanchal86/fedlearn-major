
import numpy as np
import pytest
from torch.utils.data import Dataset

from federated.partitioning.dirichlet import partition_dirichlet


class ToyDataset(Dataset):
    def __init__(self, samples=500, classes=5):
        self.labels = np.arange(samples) % classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index, int(self.labels[index])


@pytest.mark.parametrize("alpha", [0.1, 0.5, 10.0])
def test_complete_and_disjoint_partitions(alpha):
    dataset = ToyDataset()

    partitions = partition_dirichlet(
        dataset,
        num_clients=5,
        alpha=alpha,
        seed=42,
    )

    all_indices = [
        index
        for subset in partitions.values()
        for index in subset.indices
    ]

    assert len(partitions) == 5
    assert len(all_indices) == len(dataset)
    assert len(set(all_indices)) == len(dataset)
    assert set(all_indices) == set(range(len(dataset)))
    assert all(len(subset) > 0 for subset in partitions.values())


def test_reproducible_partitions():
    dataset = ToyDataset()

    first = partition_dirichlet(dataset, seed=42)
    second = partition_dirichlet(dataset, seed=42)

    for client_id in first:
        assert first[client_id].indices == second[client_id].indices


def test_different_seeds_change_partitions():
    dataset = ToyDataset()

    first = partition_dirichlet(dataset, seed=42)
    second = partition_dirichlet(dataset, seed=123)

    assert any(
        first[client_id].indices != second[client_id].indices
        for client_id in first
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_clients": 0},
        {"alpha": 0},
        {"alpha": -0.5},
        {"min_samples": 0},
        {"num_clients": 501},
    ],
)
def test_invalid_parameters(kwargs):
    dataset = ToyDataset()

    with pytest.raises(ValueError):
        partition_dirichlet(dataset, **kwargs)


def test_minimum_samples():
    dataset = ToyDataset(samples=5000, classes=5)

    partitions = partition_dirichlet(
        dataset,
        num_clients=5,
        alpha=0.5,
        seed=42,
        min_samples=10,
    )

    assert all(
        len(subset) >= 10
        for subset in partitions.values()
    )
