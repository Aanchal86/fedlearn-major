
import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.db.database import Base, get_db
from backend.app.db.models import (
    Experiment,
    TrainingRound,
    ClientParticipation,
)
from federated.contracts.experiment import ExperimentConfig


@pytest.fixture
def api_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    TestingSession = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    def override_get_db():
        with TestingSession() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client, TestingSession
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def test_results_for_new_experiment(api_client):
    client, _ = api_client

    response = client.post(
        "/api/experiments",
        json={"name": "New Federated Experiment"},
    )

    assert response.status_code == 201

    experiment_id = response.json()["id"]

    results = client.get(
        f"/api/experiments/{experiment_id}/results"
    )

    assert results.status_code == 200

    data = results.json()

    assert data["experiment_id"] == experiment_id
    assert data["status"] == "created"
    assert data["total_rounds_completed"] == 0
    assert data["history"] == []


def test_results_for_missing_experiment(api_client):
    client, _ = api_client

    response = client.get(
        "/api/experiments/999/results"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Experiment not found"


def test_results_with_saved_rounds_and_clients(api_client):
    client, TestingSession = api_client

    with TestingSession() as db:
        experiment = Experiment(
            name="Completed Federated Experiment",
            status="completed",
            configuration=ExperimentConfig().model_dump(),
        )

        first_round = TrainingRound(
            round_number=1,
            global_accuracy=0.82,
            global_loss=0.48,
            global_macro_f1=0.81,
            round_time_seconds=12.5,
        )

        first_round.clients.append(
            ClientParticipation(
                client_id="client_1",
                num_samples=100,
                aggregation_weight=0.4,
                training_loss=0.52,
                training_accuracy=0.79,
            )
        )

        first_round.clients.append(
            ClientParticipation(
                client_id="client_2",
                num_samples=150,
                aggregation_weight=0.6,
                training_loss=0.43,
                training_accuracy=0.86,
            )
        )

        second_round = TrainingRound(
            round_number=2,
            global_accuracy=0.89,
            global_loss=0.31,
            global_macro_f1=0.88,
            round_time_seconds=11.8,
        )

        experiment.rounds.extend(
            [first_round, second_round]
        )

        db.add(experiment)
        db.commit()
        experiment_id = experiment.id

    response = client.get(
        f"/api/experiments/{experiment_id}/results"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["total_rounds_completed"] == 2
    assert len(data["history"]) == 2

    round_one = data["history"][0]
    round_two = data["history"][1]

    assert round_one["round_number"] == 1
    assert round_one["global_accuracy"] == pytest.approx(0.82)
    assert round_one["global_macro_f1"] == pytest.approx(0.81)

    assert len(round_one["clients"]) == 2

    assert round_one["clients"][0]["client_id"] == "client_1"
    assert round_one["clients"][0]["num_samples"] == 100
    assert round_one["clients"][0]["aggregation_weight"] == pytest.approx(0.4)

    assert round_one["clients"][1]["client_id"] == "client_2"
    assert round_one["clients"][1]["aggregation_weight"] == pytest.approx(0.6)

    assert round_two["round_number"] == 2
    assert round_two["global_accuracy"] == pytest.approx(0.89)
    assert round_two["clients"] == []
