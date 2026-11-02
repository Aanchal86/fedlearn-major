
import pytest
import torch
from torch.utils.data import TensorDataset

from federated.networking import dataset_provider
from federated.partitioning.label_noise import NoisyLabelDataset


@pytest.fixture
def mock_dataset(monkeypatch):
    images = torch.randn(40, 1, 28, 28)
    labels = torch.tensor([0, 1] * 20)

    dataset = TensorDataset(images, labels)

    class MockSplits:
        train = dataset
        val = dataset
        test = dataset

    monkeypatch.setattr(
        dataset_provider,
        "get_dataset_splits",
        lambda dataset_name, seed: MockSplits(),
    )

    return dataset


def test_iid_client_partition(mock_dataset):
    client = dataset_provider.get_client_dataset(
        dataset_name="pneumonia_mnist",
        client_id="client_1",
        num_clients=4,
        partition_type="iid",
    )

    assert len(client) == 10


def test_iid_partition_reproducibility(mock_dataset):
    first = dataset_provider.get_client_dataset(
        "pneumonia_mnist",
        "client_1",
        num_clients=4,
        seed=42,
    )

    second = dataset_provider.get_client_dataset(
        "pneumonia_mnist",
        "client_1",
        num_clients=4,
        seed=42,
    )

    assert first.indices == second.indices


def test_different_clients_have_disjoint_samples(mock_dataset):
    first = dataset_provider.get_client_dataset(
        "pneumonia_mnist",
        "client_1",
        num_clients=4,
    )

    second = dataset_provider.get_client_dataset(
        "pneumonia_mnist",
        "client_2",
        num_clients=4,
    )

    assert set(first.indices).isdisjoint(second.indices)


def test_dirichlet_partition(mock_dataset):
    client = dataset_provider.get_client_dataset(
        "pneumonia_mnist",
        "client_1",
        num_clients=4,
        partition_type="dirichlet",
        dirichlet_alpha=1.0,
    )

    assert len(client) > 0


def test_noisy_client_partition(mock_dataset):
    client = dataset_provider.get_client_dataset(
        "pneumonia_mnist",
        "client_2",
        num_clients=4,
        noise_rate=0.5,
        noisy_client_id=2,
    )

    assert isinstance(client, NoisyLabelDataset)
    assert client.num_corrupted == 5


def test_other_client_remains_clean(mock_dataset):
    client = dataset_provider.get_client_dataset(
        "pneumonia_mnist",
        "client_1",
        num_clients=4,
        noise_rate=0.5,
        noisy_client_id=2,
    )

    assert not isinstance(client, NoisyLabelDataset)


@pytest.mark.parametrize(
    "client_id",
    ["invalid", "client_0", "client_5", "client_abc"],
)
def test_invalid_client_id(mock_dataset, client_id):
    with pytest.raises(ValueError):
        dataset_provider.get_client_dataset(
            "pneumonia_mnist",
            client_id,
            num_clients=4,
        )


def test_invalid_partition_type(mock_dataset):
    with pytest.raises(ValueError):
        dataset_provider.get_client_dataset(
            "pneumonia_mnist",
            "client_1",
            partition_type="unknown",
        )


def test_invalid_noise_configuration(mock_dataset):
    with pytest.raises(ValueError):
        dataset_provider.get_client_dataset(
            "pneumonia_mnist",
            "client_1",
            noise_rate=0.5,
            noisy_client_id=None,
        )
