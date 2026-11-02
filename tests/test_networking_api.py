
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from federated.networking.registry import (
    ClientRegistry,
)
from backend.app.api import networking


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        networking,
        "client_registry",
        ClientRegistry(),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_register_client(client):
    response = client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    assert response.status_code == 201
    assert response.json() == {
        "client_id": "client_1",
        "status": "registered",
    }


def test_duplicate_registration(client):
    client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    response = client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    assert response.status_code == 409


def test_list_registered_clients(client):
    client.post(
        "/api/network/clients/register",
        json={"client_id": "client_2"},
    )

    client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    response = client.get("/api/network/clients")

    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 2
    assert [
        item["client_id"] for item in body["clients"]
    ] == ["client_1", "client_2"]


def test_heartbeat(client):
    client.post(
        "/api/network/clients/register",
        json={"client_id": "client_1"},
    )

    response = client.post(
        "/api/network/clients/client_1/heartbeat"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "active"


def test_unknown_client_heartbeat(client):
    response = client.post(
        "/api/network/clients/unknown/heartbeat"
    )

    assert response.status_code == 404


def test_invalid_registration(client):
    response = client.post(
        "/api/network/clients/register",
        json={"client_id": ""},
    )

    assert response.status_code == 422
