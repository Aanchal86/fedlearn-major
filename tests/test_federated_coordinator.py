
import copy

import pytest
import torch
from torch.utils.data import TensorDataset

from federated.contracts.experiment import ExperimentConfig
from federated.contracts.round_result import RoundResult
from federated.server.coordinator import FederatedCoordinator


@pytest.fixture
def small_dataset(monkeypatch):
    torch.manual_seed(42)

    train = TensorDataset(
        torch.randn(30, 1, 28, 28),
        torch.randint(0, 2, (30,)),
    )

    val = TensorDataset(
        torch.randn(10, 1, 28, 28),
        torch.randint(0, 2, (10,)),
    )

    test = TensorDataset(
        torch.randn(10, 1, 28, 28),
        torch.randint(0, 2, (10,)),
    )

    from federated.data.splits import DatasetSplits

    def fake_splits(dataset_name, seed=42):
        return DatasetSplits(
            train=train,
            val=val,
            test=test,
        )

    monkeypatch.setattr(
        "federated.server.coordinator.get_dataset_splits",
        fake_splits,
    )


def make_config(**overrides):
    settings = {
        "dataset": "pneumonia_mnist",
        "num_clients": 3,
        "num_rounds": 2,
        "local_epochs": 1,
        "batch_size": 5,
        "learning_rate": 0.001,
        "random_seed": 42,
    }
    settings.update(overrides)
    return ExperimentConfig(**settings)


def test_multiple_rounds_complete(small_dataset):
    coordinator = FederatedCoordinator(make_config())

    history = coordinator.run()

    assert len(history) == 2
    assert all(isinstance(r, RoundResult) for r in history)
    assert [r.round_number for r in history] == [1, 2]


def test_global_model_weights_change(small_dataset):
    coordinator = FederatedCoordinator(make_config())

    initial = copy.deepcopy(coordinator.global_model.state_dict())

    coordinator.run()

    assert any(
        not torch.equal(
            coordinator.global_model.state_dict()[name],
            initial[name],
        )
        for name in initial
    )


def test_round_metrics_are_valid(small_dataset):
    coordinator = FederatedCoordinator(make_config())

    history = coordinator.run()

    for result in history:
        assert 0 <= result.global_accuracy <= 1
        assert result.global_loss >= 0
        assert 0 <= result.global_macro_f1 <= 1
        assert result.round_time_seconds >= 0
        assert result.aggregation_method == "fedavg"


def test_client_participation_and_weights(small_dataset):
    coordinator = FederatedCoordinator(make_config())

    history = coordinator.run()

    for result in history:
        assert result.total_samples == 30
        assert len(result.participating_clients) == 3
        assert set(result.client_weights) == {
            "client_1",
            "client_2",
            "client_3",
        }
        assert sum(result.client_weights.values()) == pytest.approx(1.0)
        assert all(
            weight == pytest.approx(1 / 3)
            for weight in result.client_weights.values()
        )


def test_invalid_aggregation_rejected(small_dataset):
    config = make_config(aggregation_method="adaptive")

    with pytest.raises(ValueError, match="FedAvg"):
        FederatedCoordinator(config)


def test_invalid_partition_rejected(small_dataset):
    config = make_config(partition_type="dirichlet")

    with pytest.raises(ValueError, match="IID"):
        FederatedCoordinator(config)


def test_deterministic_results(small_dataset):
    config = make_config()

    first = FederatedCoordinator(config).run()
    second = FederatedCoordinator(config).run()

    for a, b in zip(first, second):
        assert a.global_accuracy == pytest.approx(b.global_accuracy)
        assert a.global_loss == pytest.approx(b.global_loss, abs=1e-5)
        assert a.global_macro_f1 == pytest.approx(b.global_macro_f1)
