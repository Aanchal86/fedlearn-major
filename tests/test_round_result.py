
import pytest
from pydantic import ValidationError

from federated.contracts.round_result import RoundResult


def make_result(**overrides):
    data = {
        "round_number": 1,
        "global_accuracy": 0.82,
        "global_loss": 0.48,
        "global_macro_f1": 0.81,
        "participating_clients": [
            "client_1",
            "client_2"
        ],
        "total_samples": 24000,
        "aggregation_method": "fedavg",
        "round_time_seconds": 12.5,
        "client_weights": {
            "client_1": 0.5,
            "client_2": 0.5
        }
    }

    data.update(overrides)
    return RoundResult(**data)


def test_valid_round_result():
    result = make_result()
    assert result.global_accuracy == 0.82
    assert result.round_number == 1


def test_invalid_accuracy():
    with pytest.raises(ValidationError):
        make_result(global_accuracy=1.5)


def test_invalid_round_number():
    with pytest.raises(ValidationError):
        make_result(round_number=0)


def test_invalid_weights_sum():
    with pytest.raises(ValidationError):
        make_result(
            client_weights={
                "client_1": 0.7,
                "client_2": 0.7
            }
        )


def test_duplicate_clients():
    with pytest.raises(ValidationError):
        make_result(
            participating_clients=[
                "client_1",
                "client_1"
            ]
        )


def test_missing_client_weight():
    with pytest.raises(ValidationError):
        make_result(
            client_weights={
                "client_1": 1.0
            }
        )
