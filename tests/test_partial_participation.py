
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from federated.contracts.experiment import ExperimentConfig
from federated.networking.exchange import TrainingExchange
from federated.networking.registry import ClientRegistry
from federated.server.remote_coordinator import RemoteFederatedCoordinator


def make_coordinator(
    num_clients=2,
    min_participation_rate=0.5,
    timeout_seconds=0.02,
):
    config = ExperimentConfig(
        dataset="pneumonia_mnist",
        num_clients=num_clients,
        num_rounds=1,
        partition_type="iid",
        random_seed=42,
        device="cpu",
    )

    return RemoteFederatedCoordinator(
        config=config,
        experiment_id=19003,
        exchange=TrainingExchange(),
        registry=ClientRegistry(),
        timeout_seconds=timeout_seconds,
        poll_interval=0.005,
        min_participation_rate=min_participation_rate,
    )


def make_update(client_id):
    return SimpleNamespace(
        client_id=client_id,
        experiment_id=19003,
        round_number=1,
    )


def test_required_client_count():
    coordinator = make_coordinator(
        num_clients=5,
        min_participation_rate=0.6,
    )

    assert coordinator.required_client_count() == 3


def test_required_count_rounds_up():
    coordinator = make_coordinator(
        num_clients=3,
        min_participation_rate=0.67,
    )

    assert coordinator.required_client_count() == 3


def test_invalid_participation_rate():
    with pytest.raises(ValueError):
        make_coordinator(min_participation_rate=0)

    with pytest.raises(ValueError):
        make_coordinator(min_participation_rate=1.5)

    with pytest.raises(ValueError):
        make_coordinator(min_participation_rate=float("nan"))


def test_partial_participation_succeeds():
    coordinator = make_coordinator()
    clients = coordinator.client_ids()

    valid_update = make_update("client_1")

    with (
        patch.object(
            coordinator.exchange,
            "get_update",
            side_effect=lambda client_id, *_: (
                valid_update
                if client_id == "client_1"
                else None
            ),
        ),
        patch.object(
            coordinator,
            "validate_update",
            return_value=valid_update,
        ),
    ):
        updates = coordinator.collect_updates(
            round_number=1,
            clients=clients,
        )

    assert updates == [valid_update]

    participation = coordinator.round_participation[1]

    assert participation["received_clients"] == ["client_1"]
    assert participation["missing_clients"] == ["client_2"]
    assert participation["required_clients"] == 1
    assert participation["participation_rate"] == 0.5


def test_insufficient_participation_fails():
    coordinator = make_coordinator(
        num_clients=2,
        min_participation_rate=1.0,
    )

    with pytest.raises(
        TimeoutError,
        match="Insufficient client updates",
    ):
        coordinator.collect_updates(
            round_number=1,
            clients=coordinator.client_ids(),
        )


def test_all_clients_respond():
    coordinator = make_coordinator()
    clients = coordinator.client_ids()

    updates = {
        "client_1": make_update("client_1"),
        "client_2": make_update("client_2"),
    }

    with (
        patch.object(
            coordinator.exchange,
            "get_update",
            side_effect=lambda client_id, *_: updates[client_id],
        ),
        patch.object(
            coordinator,
            "validate_update",
            side_effect=lambda message, _: message,
        ),
    ):
        collected = coordinator.collect_updates(
            round_number=1,
            clients=clients,
        )

    assert len(collected) == 2
    assert collected[0].client_id == "client_1"
    assert collected[1].client_id == "client_2"

    participation = coordinator.round_participation[1]

    assert participation["missing_clients"] == []
    assert participation["participation_rate"] == 1.0


def test_mismatched_client_id_rejected():
    coordinator = make_coordinator()

    wrong_update = make_update("client_2")

    with patch.object(
        coordinator.exchange,
        "get_update",
        side_effect=lambda client_id, *_: (
            wrong_update
            if client_id == "client_1"
            else None
        ),
    ):
        with pytest.raises(ValueError, match="Client ID mismatch"):
            coordinator.collect_updates(
                round_number=1,
                clients=coordinator.client_ids(),
            )
