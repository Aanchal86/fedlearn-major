import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from federated.contracts.client_update import ClientUpdate
from federated.networking.exchange import TrainingExchange
from federated.networking.protocol import (
    ClientUpdateMessage,
    decode_model_payload,
    encode_model_payload,
)
from federated.networking.serialization import (
    deserialize_model_state,
    serialize_model_state,
)
from federated.server.remote_coordinator import (
    RemoteFederatedCoordinator,
)


EXPERIMENT_ID = 20004
NUM_ROUNDS = 3
CLIENT_IDS = ("client_1", "client_2")


def make_coordinator():
    coordinator = RemoteFederatedCoordinator.__new__(
        RemoteFederatedCoordinator
    )

    coordinator.config = SimpleNamespace(
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        aggregation_method="fedavg",
        device="cpu",
        num_rounds=NUM_ROUNDS,
    )

    coordinator.experiment_id = EXPERIMENT_ID
    coordinator.exchange = TrainingExchange()
    coordinator.registry = MagicMock()
    coordinator.registry.get.return_value = object()
    coordinator.global_model = torch.nn.Linear(2, 1)

    coordinator.partitions = {
        0: [0, 1],
        1: [2, 3],
    }

    coordinator.val_loader = None
    coordinator.history = []
    coordinator.client_history = {}
    coordinator.round_participation = {}
    coordinator.timeout_seconds = 5.0
    coordinator.poll_interval = 0.005
    coordinator.min_participation_rate = 1.0

    return coordinator


def copy_state(state):
    return OrderedDict(
        (name, value.detach().clone())
        for name, value in state.items()
    )


def assert_same_state(actual, expected):
    assert set(actual) == set(expected)

    for name in expected:
        torch.testing.assert_close(
            actual[name],
            expected[name],
            rtol=0,
            atol=1e-6,
        )


def run_simulated_client(coordinator, client_id, errors, records):
    try:
        for round_number in range(1, NUM_ROUNDS + 1):
            deadline = time.monotonic() + 5.0

            assignment = None

            while time.monotonic() < deadline:
                assignment = coordinator.exchange.get_assignment(
                    client_id,
                    EXPERIMENT_ID,
                    round_number,
                )

                if assignment is not None:
                    break

                time.sleep(0.005)

            if assignment is None:
                raise TimeoutError(
                    f"{client_id} did not receive round {round_number}"
                )

            global_state = deserialize_model_state(
                decode_model_payload(
                    assignment.model_state_base64
                )
            )

            records[(client_id, round_number)] = copy_state(
                global_state
            )

            increment = (
                1.0 if client_id == "client_1" else 3.0
            )

            local_state = OrderedDict(
                (name, value + increment)
                for name, value in global_state.items()
            )

            update = ClientUpdateMessage(
                experiment_id=EXPERIMENT_ID,
                client_id=client_id,
                round_number=round_number,
                num_samples=2,
                training_loss=0.4,
                training_accuracy=0.8,
                training_time_seconds=0.1,
                model_state_base64=encode_model_payload(
                    serialize_model_state(local_state)
                ),
            )

            coordinator.exchange.submit_update(update)

    except Exception as exc:
        errors.append(exc)


def test_three_round_distributed_fedavg():
    coordinator = make_coordinator()

    initial_state = copy_state(
        coordinator.global_model.state_dict()
    )

    errors = []
    records = {}

    threads = [
        threading.Thread(
            target=run_simulated_client,
            args=(coordinator, client_id, errors, records),
            daemon=True,
        )
        for client_id in CLIENT_IDS
    ]

    with patch(
        "federated.server.remote_coordinator.evaluate_model",
        return_value={
            "accuracy": 0.8,
            "loss": 0.3,
            "macro_f1": 0.75,
        },
    ):
        for thread in threads:
            thread.start()

        try:
            results = coordinator.run()
        finally:
            for thread in threads:
                thread.join(timeout=6.0)

    assert not errors, f"Client failures: {errors}"
    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == NUM_ROUNDS

    for round_number in range(1, NUM_ROUNDS + 1):
        expected_start = OrderedDict(
            (
                name,
                value + 2.0 * (round_number - 1),
            )
            for name, value in initial_state.items()
        )

        for client_id in CLIENT_IDS:
            received = records[(client_id, round_number)]

            assert_same_state(
                received,
                expected_start,
            )

        assert results[round_number - 1].round_number == round_number
        assert results[round_number - 1].total_samples == 4
        assert set(
            results[round_number - 1].participating_clients
        ) == set(CLIENT_IDS)

    expected_final = OrderedDict(
        (name, value + 2.0 * NUM_ROUNDS)
        for name, value in initial_state.items()
    )

    assert_same_state(
        coordinator.global_model.state_dict(),
        expected_final,
    )

    assert len(coordinator.history) == NUM_ROUNDS
    assert len(coordinator.round_participation) == NUM_ROUNDS

    for round_number in range(1, NUM_ROUNDS + 1):
        participation = coordinator.round_participation[
            round_number
        ]

        assert participation["participation_rate"] == 1.0
        assert participation["missing_clients"] == []


def test_round_updates_are_isolated():
    exchange = TrainingExchange()

    from federated.networking.protocol import TrainingAssignment

    for round_number in (1, 2):
        exchange.publish_assignment(
            "client_1",
            TrainingAssignment(
                experiment_id=EXPERIMENT_ID,
                round_number=round_number,
                dataset="pneumonia_mnist",
                local_epochs=1,
                batch_size=4,
                learning_rate=0.001,
                model_state_base64="bW9kZWw=",
            ),
        )

    for round_number in (1, 2):
        update = ClientUpdateMessage(
            experiment_id=EXPERIMENT_ID,
            client_id="client_1",
            round_number=round_number,
            num_samples=2,
            training_loss=0.4,
            training_accuracy=0.8,
            training_time_seconds=0.1,
            model_state_base64="bW9kZWw=",
        )

        assert exchange.submit_update(update) is True

    first = exchange.get_update(
        "client_1",
        EXPERIMENT_ID,
        1,
    )

    second = exchange.get_update(
        "client_1",
        EXPERIMENT_ID,
        2,
    )

    assert first.round_number == 1
    assert second.round_number == 2
    assert len(exchange._updates) == 2
