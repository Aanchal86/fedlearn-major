
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.api import remote_training


def make_payload(experiment_id=19004, rate=0.5):
    return {
        "experiment_id": experiment_id,
        "dataset": "pneumonia_mnist",
        "num_clients": 2,
        "num_rounds": 1,
        "min_participation_rate": rate,
        "timeout_seconds": 30,
    }


def test_start_passes_participation_rate(monkeypatch):
    monkeypatch.setenv("FEDLEARN_ENABLE_DEV_API", "1")

    coordinator = MagicMock()
    coordinator.required_client_count.return_value = 1

    with (
        patch.object(
            remote_training,
            "RemoteFederatedCoordinator",
            return_value=coordinator,
        ) as mock_coordinator,
        patch.object(
            remote_training.executor,
            "submit",
        ),
    ):
        with remote_training.jobs_lock:
            remote_training.jobs.pop(19004, None)

        response = TestClient(app).post(
            "/api/network/dev/remote/start",
            json=make_payload(),
        )

        assert response.status_code == 200
        assert response.json()["required_clients"] == 1

        assert (
            mock_coordinator.call_args.kwargs[
                "min_participation_rate"
            ]
            == 0.5
        )

    with remote_training.jobs_lock:
        remote_training.jobs.pop(19004, None)


def test_invalid_participation_rate(monkeypatch):
    monkeypatch.setenv("FEDLEARN_ENABLE_DEV_API", "1")

    with TestClient(app) as client:
        for rate in [0, -0.5, 1.5]:
            response = client.post(
                "/api/network/dev/remote/start",
                json=make_payload(
                    experiment_id=19005,
                    rate=rate,
                ),
            )

            assert response.status_code == 422


def test_default_participation_rate():
    payload = remote_training.StartRemoteTrainingRequest(
        experiment_id=19006,
    )

    assert payload.min_participation_rate == 1.0


def test_status_includes_participation(monkeypatch):
    monkeypatch.setenv("FEDLEARN_ENABLE_DEV_API", "1")

    experiment_id = 19007

    with remote_training.jobs_lock:
        remote_training.jobs[experiment_id] = {
            "status": "completed",
            "results": [],
            "error": None,
            "min_participation_rate": 0.5,
            "participation": {
                1: {
                    "expected_clients": [
                        "client_1",
                        "client_2",
                    ],
                    "received_clients": ["client_1"],
                    "missing_clients": ["client_2"],
                    "required_clients": 1,
                    "participation_rate": 0.5,
                }
            },
        }

    try:
        response = TestClient(app).get(
            f"/api/network/dev/remote/status/{experiment_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "completed"
        assert data["participation"]["1"]["missing_clients"] == [
            "client_2"
        ]
    finally:
        with remote_training.jobs_lock:
            remote_training.jobs.pop(experiment_id, None)


def test_dev_api_disabled(monkeypatch):
    monkeypatch.delenv(
        "FEDLEARN_ENABLE_DEV_API",
        raising=False,
    )

    response = TestClient(app).post(
        "/api/network/dev/remote/start",
        json=make_payload(experiment_id=19008),
    )

    assert response.status_code == 404
