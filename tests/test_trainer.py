
import pytest
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated.models.cnn import SmallCNN
from federated.training.trainer import (
    train_model,
    train_one_epoch,
)


@pytest.fixture
def small_loader():
    images = torch.randn(12, 1, 28, 28)
    labels = torch.randint(0, 2, (12,))

    dataset = TensorDataset(images, labels)

    return DataLoader(dataset, batch_size=4)


def test_train_one_epoch_returns_metrics(small_loader):
    model = SmallCNN(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    metrics = train_one_epoch(
        model=model,
        train_loader=small_loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert metrics["num_samples"] == 12
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["loss"] >= 0
    assert metrics["duration_seconds"] >= 0


def test_training_updates_weights(small_loader):
    model = SmallCNN(num_classes=2)

    original_weights = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    train_model(
        model=model,
        train_loader=small_loader,
        epochs=1,
        device="cpu",
    )

    assert any(
        not torch.equal(
            original_weights[name],
            parameter.detach(),
        )
        for name, parameter in model.named_parameters()
    )


def test_multiple_epochs(small_loader):
    model = SmallCNN(num_classes=2)

    history = train_model(
        model=model,
        train_loader=small_loader,
        epochs=2,
        device="cpu",
    )

    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[1]["epoch"] == 2


def test_invalid_epochs(small_loader):
    model = SmallCNN(num_classes=2)

    with pytest.raises(ValueError):
        train_model(
            model=model,
            train_loader=small_loader,
            epochs=0,
        )


def test_invalid_learning_rate(small_loader):
    model = SmallCNN(num_classes=2)

    with pytest.raises(ValueError):
        train_model(
            model=model,
            train_loader=small_loader,
            learning_rate=0,
        )


def test_empty_training_loader():
    model = SmallCNN(num_classes=2)

    images = torch.empty(0, 1, 28, 28)
    labels = torch.empty(0, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )

    optimizer = torch.optim.Adam(model.parameters())

    with pytest.raises(ValueError):
        train_one_epoch(
            model=model,
            train_loader=loader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
        )
