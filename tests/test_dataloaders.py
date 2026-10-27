
import pytest
import torch

from federated.data.dataloaders import (
    create_dataloaders,
)


@pytest.mark.parametrize(
    "dataset_name,num_classes",
    [
        ("fashion_mnist", 10),
        ("pneumonia_mnist", 2),
    ],
)
def test_dataloader_batch_format(
    dataset_name,
    num_classes,
):
    train_loader, _, _ = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=64,
        seed=42,
    )

    images, labels = next(iter(train_loader))

    assert images.shape == (64, 1, 28, 28)
    assert labels.shape == (64,)

    assert images.dtype == torch.float32
    assert labels.dtype == torch.long

    assert torch.isfinite(images).all()

    assert labels.min() >= 0
    assert labels.max() < num_classes


@pytest.mark.parametrize(
    "dataset_name",
    ["fashion_mnist", "pneumonia_mnist"],
)
def test_dataloader_dataset_sizes(dataset_name):
    train_loader, val_loader, test_loader = (
        create_dataloaders(dataset_name)
    )

    expected = {
        "fashion_mnist": (54000, 6000, 10000),
        "pneumonia_mnist": (4708, 524, 624),
    }

    train_size, val_size, test_size = expected[
        dataset_name
    ]

    assert len(train_loader.dataset) == train_size
    assert len(val_loader.dataset) == val_size
    assert len(test_loader.dataset) == test_size


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        create_dataloaders(
            "fashion_mnist",
            batch_size=0,
        )


def test_reproducible_training_batches():
    first, _, _ = create_dataloaders(
        "fashion_mnist",
        seed=42,
    )

    second, _, _ = create_dataloaders(
        "fashion_mnist",
        seed=42,
    )

    first_images, first_labels = next(iter(first))
    second_images, second_labels = next(iter(second))

    assert torch.equal(first_images, second_images)
    assert torch.equal(first_labels, second_labels)
