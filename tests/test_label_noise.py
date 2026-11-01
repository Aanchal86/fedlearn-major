
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from federated.partitioning.label_noise import NoisyLabelDataset


@pytest.fixture
def sample_dataset():
    images = torch.randn(100, 1, 28, 28)
    labels = torch.arange(100) % 10
    return TensorDataset(images, labels)


def test_noise_changes_correct_number_of_labels(sample_dataset):
    noisy = NoisyLabelDataset(
        sample_dataset,
        noise_rate=0.3,
        num_classes=10,
        seed=42,
    )

    changed = sum(
        int(sample_dataset[i][1]) != int(noisy[i][1])
        for i in range(len(sample_dataset))
    )

    assert changed == 30
    assert noisy.num_corrupted == 30


def test_noise_is_reproducible(sample_dataset):
    first = NoisyLabelDataset(
        sample_dataset, 0.3, 10, seed=42
    )
    second = NoisyLabelDataset(
        sample_dataset, 0.3, 10, seed=42
    )

    assert first.corrupted_labels == second.corrupted_labels


def test_zero_noise_preserves_labels(sample_dataset):
    noisy = NoisyLabelDataset(
        sample_dataset, 0.0, 10, seed=42
    )

    assert noisy.num_corrupted == 0

    for i in range(len(sample_dataset)):
        assert int(noisy[i][1]) == int(sample_dataset[i][1])


def test_original_dataset_unchanged(sample_dataset):
    labels_before = [
        int(sample_dataset[i][1])
        for i in range(len(sample_dataset))
    ]

    NoisyLabelDataset(
        sample_dataset, 0.5, 10, seed=42
    )

    labels_after = [
        int(sample_dataset[i][1])
        for i in range(len(sample_dataset))
    ]

    assert labels_before == labels_after


@pytest.mark.parametrize("noise_rate", [-0.1, 1.1])
def test_invalid_noise_rate(sample_dataset, noise_rate):
    with pytest.raises(ValueError):
        NoisyLabelDataset(
            sample_dataset,
            noise_rate=noise_rate,
            num_classes=10,
        )


def test_binary_class_noise():
    images = torch.randn(20, 1, 28, 28)
    labels = torch.arange(20) % 2

    dataset = TensorDataset(images, labels)

    noisy = NoisyLabelDataset(
        dataset,
        noise_rate=1.0,
        num_classes=2,
        seed=42,
    )

    assert all(
        int(dataset[i][1]) != int(noisy[i][1])
        for i in range(len(dataset))
    )


def test_dataset_length_preserved(sample_dataset):
    noisy = NoisyLabelDataset(
        sample_dataset, 0.3, 10, seed=42
    )

    assert len(noisy) == len(sample_dataset)


def test_noisy_labels_work_with_dataloader():
    from torch.utils.data import DataLoader

    images = torch.randn(100, 1, 28, 28)
    labels = torch.arange(100) % 10
    dataset = TensorDataset(images, labels)

    noisy = NoisyLabelDataset(
        dataset,
        noise_rate=0.3,
        num_classes=10,
        seed=42,
    )

    loader = DataLoader(
        noisy,
        batch_size=16,
        shuffle=False,
    )

    for images_batch, labels_batch in loader:
        assert isinstance(labels_batch, torch.Tensor)
        assert labels_batch.dtype == labels.dtype
        assert len(images_batch) == len(labels_batch)
