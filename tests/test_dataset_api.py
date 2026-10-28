
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_list_datasets():
    response = client.get("/api/datasets")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 2

    names = {item["name"] for item in data}

    assert names == {
        "fashion_mnist",
        "pneumonia_mnist",
    }


def test_get_fashion_dataset():
    response = client.get(
        "/api/datasets/fashion_mnist"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["num_classes"] == 10
    assert data["train_samples"] == 54000


def test_get_pneumonia_dataset():
    response = client.get(
        "/api/datasets/pneumonia_mnist"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["num_classes"] == 2
    assert data["train_samples"] == 4708


def test_unknown_dataset():
    response = client.get(
        "/api/datasets/unknown"
    )

    assert response.status_code == 404
