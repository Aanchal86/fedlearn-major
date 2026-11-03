from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.app.api.networking import submit_client_update
from federated.networking.exchange import TrainingExchange
from federated.networking.http_client import FederatedHTTPClient
from federated.networking.protocol import (
    ClientUpdateMessage,
    TrainingAssignment,
)


@pytest.fixture
def exchange():
    return TrainingExchange()


@pytest.fixture
def assignment():
    return TrainingAssignment(
        experiment_id=20003,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        model_state_base64="bW9kZWw=",
    )


@pytest.fixture
def update():
    return ClientUpdateMessage(
        experiment_id=20003,
        client_id="client_1",
        round_number=1,
        num_samples=2,
        training_loss=0.4,
        training_accuracy=0.8,
        training_time_seconds=0.1,
        model_state_base64="bW9kZWw=",
    )


def test_first_submission_accepted(exchange, assignment, update):
    exchange.publish_assignment("client_1", assignment)

    assert exchange.submit_update(update) is True
    assert exchange.get_update("client_1", 20003, 1) == update


def test_identical_retry_is_safe(exchange, assignment, update):
    exchange.publish_assignment("client_1", assignment)

    assert exchange.submit_update(update) is True
    assert exchange.submit_update(update) is False

    assert len(exchange._updates) == 1


def test_conflicting_retry_rejected(exchange, assignment, update):
    exchange.publish_assignment("client_1", assignment)
    exchange.submit_update(update)

    changed = update.model_copy(
        update={"training_loss": 0.9}
    )

    with pytest.raises(
        ValueError,
        match="Conflicting update",
    ):
        exchange.submit_update(changed)


def test_update_without_assignment_rejected(exchange, update):
    with pytest.raises(KeyError):
        exchange.submit_update(update)


def test_same_client_can_submit_next_round(
    exchange, assignment, update
):
    exchange.publish_assignment("client_1", assignment)
    assert exchange.submit_update(update) is True

    next_assignment = assignment.model_copy(
        update={"round_number": 2}
    )
    next_update = update.model_copy(
        update={"round_number": 2}
    )

    exchange.publish_assignment(
        "client_1",
        next_assignment,
    )

    assert exchange.submit_update(next_update) is True
    assert len(exchange._updates) == 2


def test_api_returns_already_accepted(
    exchange, assignment, update
):
    exchange.publish_assignment("client_1", assignment)

    with (
        patch(
            "backend.app.api.networking.training_exchange",
            exchange,
        ),
        patch(
            "backend.app.api.networking.client_registry"
        ) as registry,
    ):
        registry.get.return_value = object()

        first = submit_client_update("client_1", update)
        second = submit_client_update("client_1", update)

    assert first["status"] == "accepted"
    assert second["status"] == "already_accepted"


def test_http_client_retries_after_connection_error(update):
    client = FederatedHTTPClient(
        client_id="client_1",
        max_retries=1,
        backoff_factor=0,
    )

    request = httpx.Request(
        "POST",
        "http://127.0.0.1:8000/api/network/clients/client_1/updates",
    )

    success = httpx.Response(
        200,
        request=request,
        json={"status": "already_accepted"},
    )

    mock_http = MagicMock()
    mock_http.request.side_effect = [
        httpx.ConnectError("Connection lost", request=request),
        success,
    ]

    context = MagicMock()
    context.__enter__.return_value = mock_http

    with patch(
        "federated.networking.http_client.httpx.Client",
        return_value=context,
    ):
        result = client.submit_update(update)

    assert result["status"] == "already_accepted"
    assert mock_http.request.call_count == 2


def test_http_client_rejects_wrong_client(update):
    client = FederatedHTTPClient(
        client_id="client_2",
    )

    with pytest.raises(
        ValueError,
        match="Client ID mismatch",
    ):
        client.submit_update(update)
