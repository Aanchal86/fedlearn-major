
from collections import OrderedDict

import pytest
import torch

from federated.contracts.experiment import ExperimentConfig
from federated.networking.exchange import TrainingExchange
from federated.networking.registry import ClientRegistry
from federated.networking.protocol import (
    ClientUpdateMessage,
    encode_model_payload,
)
from federated.networking.serialization import (
    serialize_model_state,
)
from federated.server.remote_coordinator import (
    RemoteFederatedCoordinator,
)


@pytest.fixture
def coordinator():
    config = ExperimentConfig(
        dataset="pneumonia_mnist",
        num_clients=2,
        num_rounds=1,
        local_epochs=1,
        batch_size=64,
        learning_rate=0.001,
        aggregation_method="fedavg",
        partition_type="iid",
        random_seed=42,
        device="cpu",
    )

    return RemoteFederatedCoordinator(
        config=config,
        experiment_id=18004,
        exchange=TrainingExchange(),
        registry=ClientRegistry(),
        timeout_seconds=2,
        poll_interval=0.01,
    )


def make_update(coordinator, state=None, **overrides):
    if state is None:
        state = OrderedDict(
            (key, value.detach().clone())
            for key, value
            in coordinator.global_model.state_dict().items()
        )

    data = {
        "experiment_id": 18004,
        "client_id": "client_1",
        "round_number": 1,
        "num_samples": len(coordinator.partitions[0]),
        "training_loss": 0.3,
        "training_accuracy": 0.8,
        "training_time_seconds": 1.0,
        "model_state_base64": encode_model_payload(
            serialize_model_state(state)
        ),
    }

    data.update(overrides)
    return ClientUpdateMessage(**data)


def test_valid_update_accepted(coordinator):
    message = make_update(coordinator)
    update = coordinator.validate_update(message, 1)

    assert update.client_id == "client_1"
    assert update.round_number == 1
    assert update.num_samples == len(coordinator.partitions[0])


def test_wrong_experiment_rejected(coordinator):
    message = make_update(
        coordinator,
        experiment_id=999,
    )

    with pytest.raises(ValueError):
        coordinator.validate_update(message, 1)


def test_wrong_round_rejected(coordinator):
    message = make_update(
        coordinator,
        round_number=2,
    )

    with pytest.raises(ValueError):
        coordinator.validate_update(message, 1)


def test_unknown_client_rejected(coordinator):
    message = make_update(
        coordinator,
        client_id="client_999",
    )

    with pytest.raises(ValueError, match="Unknown client ID"):
        coordinator.validate_update(message, 1)


def test_incorrect_sample_count_rejected(coordinator):
    expected = len(coordinator.partitions[0])

    message = make_update(
        coordinator,
        num_samples=expected + 1000,
    )

    with pytest.raises(ValueError, match="Sample count mismatch"):
        coordinator.validate_update(message, 1)


def test_missing_model_parameter_rejected(coordinator):
    state = OrderedDict(
        (key, value.detach().clone())
        for key, value
        in coordinator.global_model.state_dict().items()
    )

    state.pop(next(iter(state)))

    message = make_update(coordinator, state=state)

    with pytest.raises(ValueError):
        coordinator.validate_update(message, 1)


def test_wrong_tensor_shape_rejected(coordinator):
    state = OrderedDict(
        (key, value.detach().clone())
        for key, value
        in coordinator.global_model.state_dict().items()
    )

    key = next(
        key
        for key, tensor in state.items()
        if tensor.ndim > 0
    )

    state[key] = state[key].reshape(-1)[:1]

    message = make_update(coordinator, state=state)

    with pytest.raises(ValueError):
        coordinator.validate_update(message, 1)


def test_wrong_tensor_dtype_rejected(coordinator):
    state = OrderedDict(
        (key, value.detach().clone())
        for key, value
        in coordinator.global_model.state_dict().items()
    )

    key = next(
        key
        for key, tensor in state.items()
        if tensor.is_floating_point()
    )

    state[key] = state[key].to(torch.float64)

    message = make_update(coordinator, state=state)

    with pytest.raises(ValueError):
        coordinator.validate_update(message, 1)


def test_non_finite_tensor_rejected(coordinator):
    state = OrderedDict(
        (key, value.detach().clone())
        for key, value
        in coordinator.global_model.state_dict().items()
    )

    key = next(
        key
        for key, tensor in state.items()
        if tensor.is_floating_point()
    )

    state[key].view(-1)[0] = float("nan")

    try:
        message = make_update(coordinator, state=state)
    except ValueError:
        return

    with pytest.raises(ValueError):
        coordinator.validate_update(message, 1)


def test_invalid_sample_count_rejected(coordinator):
    with pytest.raises(ValueError):
        message = make_update(
            coordinator,
            num_samples=0,
        )
        coordinator.validate_update(message, 1)
