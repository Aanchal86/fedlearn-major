
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
)


def test_valid_experiment_create():
    experiment = ExperimentCreate(
        name="Pneumonia FedAvg",
        config={
            "dataset": "pneumonia_mnist",
            "num_clients": 5,
        },
    )

    assert experiment.config.dataset == "pneumonia_mnist"
    assert experiment.config.num_clients == 5
    assert experiment.config.num_rounds == 10


def test_default_experiment_config():
    experiment = ExperimentCreate(
        name="Fashion Baseline"
    )

    assert experiment.config.dataset == "fashion_mnist"
    assert experiment.config.aggregation_method == "fedavg"


def test_invalid_experiment_name():
    with pytest.raises(ValidationError):
        ExperimentCreate(
            name="AB"
        )


def test_invalid_client_count():
    with pytest.raises(ValidationError):
        ExperimentCreate(
            name="Invalid Clients",
            config={
                "num_clients": 0
            },
        )


def test_unsupported_dataset():
    with pytest.raises(ValidationError):
        ExperimentCreate(
            name="Invalid Dataset",
            config={
                "dataset": "unknown_dataset"
            },
        )


def test_experiment_response():
    response = ExperimentResponse(
        id=1,
        name="Fashion Baseline",
        status="created",
        configuration={
            "dataset": "fashion_mnist"
        },
        created_at=datetime.now(timezone.utc),
    )

    assert response.id == 1
    assert response.configuration.dataset == "fashion_mnist"


def test_experiment_list_response():
    result = ExperimentListResponse(
        experiments=[],
        total=0,
    )

    assert result.total == 0
    assert result.experiments == []
