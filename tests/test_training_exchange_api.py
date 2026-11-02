
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api import networking
from federated.networking.exchange import TrainingExchange
from federated.networking.protocol import (
    TrainingAssignment,
    encode_model_payload,
)
from federated.networking.registry import ClientRegistry


@pytest.fixture
def setup_network(monkeypatch):
    registry = ClientRegistry()
    exchange = TrainingExchange()

    monkeypatch.setattr(
        networking,
        "client_registry",
        registry,
    )
    monkeypatch.setattr(
        networking,
        "training_exchange",
        exchange,
    )

    with TestClient(app) as client:
        yield client, exchange


def make_assignment():
    return TrainingAssignment(
        experiment_id=1,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        model_state_base64=encode_model_payload(b"model"),
    )


def make_update():
    return {
        "experiment_id": 1,
        "client_id": "client_1",
        "round_number": 1,
        "num_samples": 100,
        "training_loss": 0.3,
        "training_accuracy": 0.9,
        "training_time_seconds": 1.5,
        "model_state_base64": encode_model_payload(
            b"updated-model"
        ),
    }


def test_get_training_assignment(setup_network):
    client, exchange = setup_network

    client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    exchange.publish_assignment(
        "client_1",
        make_assignment(),
    )

    response = client.get(
        "/api/network/clients/client_1/assignments/1/1"
    )

    assert response.status_code == 200
    assert response.json()["round_number"] == 1


def test_submit_client_update(setup_network):
    client, exchange = setup_network

    client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    exchange.publish_assignment(
        "client_1",
        make_assignment(),
    )

    response = client.post(
        "/api/network/clients/client_1/updates",
        json=make_update(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"

    stored = exchange.get_update("client_1", 1, 1)

    assert stored is not None
    assert stored.num_samples == 100


def test_duplicate_update_rejected(setup_network):
    client, exchange = setup_network

    client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    exchange.publish_assignment(
        "client_1",
        make_assignment(),
    )

    first = client.post(
        "/api/network/clients/client_1/updates",
        json=make_update(),
    )

    second = client.post(
        "/api/network/clients/client_1/updates",
        json=make_update(),
    )

    assert first.status_code == 200
    assert second.status_code == 409


def test_unregistered_client_rejected(setup_network):
    client, _ = setup_network

    response = client.get(
        "/api/network/clients/unknown/assignments/1/1"
    )

    assert response.status_code == 404
