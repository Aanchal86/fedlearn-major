
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.db.database import Base, get_db
from backend.app.db import models


@pytest.fixture
def client():
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
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def test_create_experiment(client):
    response = client.post(
        "/api/experiments",
        json={
            "name": "Pneumonia FedAvg",
            "config": {
                "dataset": "pneumonia_mnist",
                "num_clients": 5,
            },
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["id"] == 1
    assert data["status"] == "created"
    assert data["configuration"]["dataset"] == "pneumonia_mnist"


def test_list_experiments(client):
    client.post(
        "/api/experiments",
        json={"name": "Fashion Baseline"},
    )

    response = client.get("/api/experiments")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert len(response.json()["experiments"]) == 1


def test_get_experiment(client):
    created = client.post(
        "/api/experiments",
        json={"name": "Fashion Baseline"},
    ).json()

    response = client.get(
        f"/api/experiments/{created['id']}"
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Fashion Baseline"


def test_experiment_not_found(client):
    response = client.get("/api/experiments/999")

    assert response.status_code == 404


def test_invalid_experiment(client):
    response = client.post(
        "/api/experiments",
        json={
            "name": "Invalid Experiment",
            "config": {
                "num_clients": 0,
            },
        },
    )

    assert response.status_code == 422
