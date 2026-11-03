import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from federated.contracts.client_update import ClientUpdate
from federated.networking.exchange import TrainingExchange
from federated.networking.protocol import decode_model_payload
from federated.networking.serialization import deserialize_model_state
from federated.server.remote_coordinator import RemoteFederatedCoordinator


@pytest.fixture
def coordinator():
    instance = RemoteFederatedCoordinator.__new__(
        RemoteFederatedCoordinator
    )

    instance.config = SimpleNamespace(
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        aggregation_method="fedavg",
        device="cpu",
        num_rounds=3,
    )

    instance.experiment_id = 20002
    instance.exchange = TrainingExchange()
    instance.registry = MagicMock()
    instance.registry.get.return_value = object()

    instance.global_model = torch.nn.Linear(2, 1)
    instance.partitions = {
        0: [0, 1],
        1: [2, 3],
    }

    instance.val_loader = None
    instance.history = []
    instance.client_history = {}
    instance.round_participation = {}
    instance.min_participation_rate = 1.0

    return instance


def assignment_state(coordinator, client_id, round_number):
    assignment = coordinator.exchange.get_assignment(
        client_id,
        coordinator.experiment_id,
        round_number,
    )

    assert assignment is not None

    return deserialize_model_state(
        decode_model_payload(assignment.model_state_base64)
    )


def assert_states_equal(first, second):
    assert set(first) == set(second)

    for name in first:
        torch.testing.assert_close(
            first[name],
            second[name],
            rtol=0,
            atol=0,
        )


def make_update(client_id, round_number, state, num_samples=2):
    return ClientUpdate(
        client_id=client_id,
        round_number=round_number,
        num_samples=num_samples,
        model_state=OrderedDict(
            (name, tensor.detach().clone())
            for name, tensor in state.items()
        ),
        training_loss=0.4,
        training_accuracy=0.8,
        training_time_seconds=0.1,
    )


def mock_evaluation():
    return patch(
        "federated.server.remote_coordinator.evaluate_model",
        return_value={
            "accuracy": 0.8,
            "loss": 0.3,
            "macro_f1": 0.75,
        },
    )


def test_first_round_uses_initial_global_model(coordinator):
    initial_state = OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in coordinator.global_model.state_dict().items()
    )

    coordinator.publish_round(1)

    received = assignment_state(coordinator, "client_1", 1)

    assert_states_equal(initial_state, received)


def test_all_clients_receive_identical_weights(coordinator):
    coordinator.publish_round(1)

    first = assignment_state(coordinator, "client_1", 1)
    second = assignment_state(coordinator, "client_2", 1)

    assert_states_equal(first, second)


def test_assignment_round_metadata(coordinator):
    coordinator.publish_round(2)

    for client_id in ("client_1", "client_2"):
        assignment = coordinator.exchange.get_assignment(
            client_id,
            20002,
            2,
        )

        assert assignment.experiment_id == 20002
        assert assignment.round_number == 2
        assert assignment.dataset == "pneumonia_mnist"


def test_next_round_receives_updated_global_weights(coordinator):
    coordinator.publish_round(1)

    original = assignment_state(coordinator, "client_1", 1)

    updated = OrderedDict(
        (name, tensor.detach().clone() + 1.0)
        for name, tensor in original.items()
    )

    coordinator.global_model.load_state_dict(
        updated,
        strict=True,
    )

    coordinator.publish_round(2)

    second_round = assignment_state(coordinator, "client_1", 2)

    assert_states_equal(updated, second_round)

    assert any(
        not torch.equal(original[name], second_round[name])
        for name in original
    )


def test_previous_round_assignment_is_immutable(coordinator):
    coordinator.publish_round(1)

    original = assignment_state(coordinator, "client_1", 1)

    with torch.no_grad():
        for parameter in coordinator.global_model.parameters():
            parameter.add_(5.0)

    coordinator.publish_round(2)

    old_assignment = assignment_state(coordinator, "client_1", 1)

    assert_states_equal(original, old_assignment)


def test_aggregation_updates_global_model(coordinator):
    initial = OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in coordinator.global_model.state_dict().items()
    )

    first_state = OrderedDict(
        (name, value + 1.0)
        for name, value in initial.items()
    )

    second_state = OrderedDict(
        (name, value + 3.0)
        for name, value in initial.items()
    )

    updates = [
        make_update("client_1", 1, first_state),
        make_update("client_2", 1, second_state),
    ]

    with mock_evaluation():
        result = coordinator.aggregate_round(
            round_number=1,
            updates=updates,
            round_start=time.perf_counter(),
        )

    expected = OrderedDict(
        (name, value + 2.0)
        for name, value in initial.items()
    )

    actual = coordinator.global_model.state_dict()

    for name in expected:
        torch.testing.assert_close(
            actual[name],
            expected[name],
        )

    assert result.round_number == 1
    assert result.client_weights == {
        "client_1": 0.5,
        "client_2": 0.5,
    }


def test_aggregated_model_used_in_next_assignment(coordinator):
    initial = OrderedDict(
        (name, tensor.detach().clone())
        for name, tensor in coordinator.global_model.state_dict().items()
    )

    updated = OrderedDict(
        (name, value + 2.0)
        for name, value in initial.items()
    )

    update = make_update(
        client_id="client_1",
        round_number=1,
        state=updated,
    )

    with mock_evaluation():
        coordinator.aggregate_round(
            round_number=1,
            updates=[update],
            round_start=time.perf_counter(),
        )

    coordinator.publish_round(2)

    received = assignment_state(coordinator, "client_2", 2)

    assert_states_equal(updated, received)


def test_missing_registration_blocks_publication(coordinator):
    coordinator.registry.get.side_effect = (
        lambda client_id: (
            None if client_id == "client_2" else object()
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Client not registered",
    ):
        coordinator.publish_round(1)

    assert coordinator.exchange.get_assignment(
        "client_1",
        20002,
        1,
    ) is None
