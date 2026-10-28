
import pytest
import torch

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from federated.models.cnn import SmallCNN
from federated.training.evaluator import evaluate_model


@pytest.fixture
def evaluation_loader():
    images = torch.randn(12, 1, 28, 28)
    labels = torch.tensor([
        0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1,
    ])

    return DataLoader(
        TensorDataset(images, labels),
        batch_size=4,
    )


def test_evaluation_returns_metrics(evaluation_loader):
    model = SmallCNN(num_classes=2)

    metrics = evaluate_model(model, evaluation_loader)

    assert metrics["num_samples"] == 12
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert 0 <= metrics["weighted_f1"] <= 1
    assert metrics["loss"] >= 0


def test_confusion_matrix_shape(evaluation_loader):
    model = SmallCNN(num_classes=2)

    metrics = evaluate_model(model, evaluation_loader)

    matrix = metrics["confusion_matrix"]

    assert len(matrix) == 2
    assert all(len(row) == 2 for row in matrix)
    assert sum(map(sum, matrix)) == 12


def test_per_class_metrics(evaluation_loader):
    model = SmallCNN(num_classes=2)

    metrics = evaluate_model(model, evaluation_loader)

    assert len(metrics["per_class_precision"]) == 2
    assert len(metrics["per_class_recall"]) == 2
    assert len(metrics["per_class_f1"]) == 2
    assert metrics["per_class_support"] == [6, 6]


def test_evaluation_does_not_update_weights(evaluation_loader):
    model = SmallCNN(num_classes=2)

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    evaluate_model(model, evaluation_loader)

    for name, parameter in model.named_parameters():
        assert torch.equal(
            before[name],
            parameter.detach(),
        )


def test_evaluation_sets_eval_mode(evaluation_loader):
    model = SmallCNN(num_classes=2)
    model.train()

    evaluate_model(model, evaluation_loader)

    assert model.training is False


def test_empty_evaluation_loader():
    model = SmallCNN(num_classes=2)

    loader = DataLoader(
        TensorDataset(
            torch.empty(0, 1, 28, 28),
            torch.empty(0, dtype=torch.long),
        ),
        batch_size=4,
    )

    with pytest.raises(ValueError):
        evaluate_model(model, loader)


def test_missing_class_still_has_fixed_matrix():
    model = SmallCNN(num_classes=2)

    images = torch.randn(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
    )

    metrics = evaluate_model(model, loader)

    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["per_class_recall"]) == 2
    assert metrics["per_class_support"] == [4, 0]
