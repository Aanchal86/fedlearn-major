
import pytest
import torch
from torch.utils.data import TensorDataset

from federated.partitioning.iid import (
    partition_iid,
    get_label_distribution,
)


@pytest.mark.parametrize("dataset_size", [54000, 4708])
def test_partition_sizes_and_coverage(dataset_size):
    dataset = TensorDataset(torch.arange(dataset_size))

    clients = partition_iid(dataset, num_clients=5, seed=42)

    assert len(clients) == 5

    all_indices = [
        index
        for client in clients.values()
        for index in client.indices
    ]

    assert len(all_indices) == dataset_size
    assert len(set(all_indices)) == dataset_size
    assert set(all_indices) == set(range(dataset_size))

    sizes = [len(client) for client in clients.values()]
    assert max(sizes) - min(sizes) <= 1


def test_partition_is_reproducible():
    dataset = TensorDataset(torch.arange(100))

    first = partition_iid(dataset, num_clients=5, seed=42)
    second = partition_iid(dataset, num_clients=5, seed=42)

    for client_id in range(5):
        assert first[client_id].indices == second[client_id].indices


def test_different_seeds_produce_different_partitions():
    dataset = TensorDataset(torch.arange(100))

    first = partition_iid(dataset, seed=42)
    second = partition_iid(dataset, seed=123)

    assert first[0].indices != second[0].indices


def test_invalid_client_count():
    dataset = TensorDataset(torch.arange(10))

    with pytest.raises(ValueError):
        partition_iid(dataset, num_clients=0)

    with pytest.raises(ValueError):
        partition_iid(dataset, num_clients=11)


def test_label_distribution():
    images = torch.arange(6)
    labels = torch.tensor([0, 1, 1, 2, 2, 2])
    dataset = TensorDataset(images, labels)

    distribution = get_label_distribution(dataset)

    assert distribution == {0: 1, 1: 2, 2: 3}
