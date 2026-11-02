
import httpx
import pytest

from federated.networking.http_client import (
    FederatedHTTPClient,
)
from federated.networking.protocol import (
    ClientUpdateMessage,
    encode_model_payload,
)


def make_mock_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original_client = httpx.Client

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)


def test_register_client(monkeypatch):
    def handler(request):
        assert request.method == "POST"
        assert request.url.path.endswith(
            "/clients/register"
        )

        return httpx.Response(
            201,
            json={
                "client_id": "client_1",
                "status": "registered",
            },
        )

    make_mock_client(monkeypatch, handler)

    client = FederatedHTTPClient("client_1")
    result = client.register()

    assert result["status"] == "registered"


def test_heartbeat(monkeypatch):
    def handler(request):
        assert request.method == "POST"
        assert request.url.path.endswith(
            "/client_1/heartbeat"
        )

        return httpx.Response(
            200,
            json={
                "client_id": "client_1",
                "status": "active",
            },
        )

    make_mock_client(monkeypatch, handler)

    client = FederatedHTTPClient("client_1")
    result = client.heartbeat()

    assert result["status"] == "active"


def test_get_assignment(monkeypatch):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "experiment_id": 1,
                "round_number": 1,
                "dataset": "pneumonia_mnist",
                "local_epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.001,
                "model_state_base64": encode_model_payload(
                    b"model"
                ),
            },
        )

    make_mock_client(monkeypatch, handler)

    client = FederatedHTTPClient("client_1")
    assignment = client.get_assignment(1, 1)

    assert assignment.experiment_id == 1
    assert assignment.round_number == 1


def test_submit_update(monkeypatch):
    def handler(request):
        assert request.method == "POST"
        assert request.url.path.endswith(
            "/client_1/updates"
        )

        return httpx.Response(
            200,
            json={"status": "accepted"},
        )

    make_mock_client(monkeypatch, handler)

    client = FederatedHTTPClient("client_1")

    update = ClientUpdateMessage(
        experiment_id=1,
        client_id="client_1",
        round_number=1,
        num_samples=10,
        training_loss=0.2,
        training_accuracy=0.8,
        training_time_seconds=1.0,
        model_state_base64=encode_model_payload(
            b"updated-model"
        ),
    )

    result = client.submit_update(update)

    assert result["status"] == "accepted"


def test_client_id_mismatch(monkeypatch):
    client = FederatedHTTPClient("client_1")

    update = ClientUpdateMessage(
        experiment_id=1,
        client_id="client_2",
        round_number=1,
        num_samples=10,
        training_loss=0.2,
        training_accuracy=0.8,
        training_time_seconds=1.0,
        model_state_base64=encode_model_payload(
            b"updated-model"
        ),
    )

    with pytest.raises(ValueError):
        client.submit_update(update)


def test_wait_for_assignment_timeout(monkeypatch):
    def handler(request):
        return httpx.Response(
            404,
            json={"detail": "Assignment not found"},
            request=request,
        )

    make_mock_client(monkeypatch, handler)

    client = FederatedHTTPClient("client_1")

    with pytest.raises(TimeoutError):
        client.wait_for_assignment(
            experiment_id=1,
            round_number=1,
            poll_interval=0.01,
            max_wait_seconds=0.03,
        )
