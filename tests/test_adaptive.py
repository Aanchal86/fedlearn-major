
import math
from collections import OrderedDict

import pytest
import torch

from federated.aggregation.adaptive import (
    adaptive_averaging,
    calculate_adaptive_weights,
)
from federated.contracts.client_update import ClientUpdate


def make_update(client_id, samples, accuracy, parameter):
    return ClientUpdate(
        client_id=client_id,
        round_number=1,
        num_samples=samples,
        model_state=OrderedDict({
            "weight": torch.tensor([parameter], dtype=torch.float32)
        }),
        training_loss=0.5,
        training_accuracy=accuracy,
        training_time_seconds=1.0,
    )


def test_adaptive_weights_sum_to_one():
    updates = [
        make_update("client_1", 1000, 0.9, 2.0),
        make_update("client_2", 500, 0.8, 4.0),
        make_update("client_3", 500, 0.6, 6.0),
    ]

    weights = calculate_adaptive_weights(updates)

    assert math.isclose(sum(weights.values()), 1.0)
    assert weights["client_1"] == pytest.approx(0.5625)
    assert weights["client_2"] == pytest.approx(0.25)
    assert weights["client_3"] == pytest.approx(0.1875)


def test_adaptive_aggregated_parameter():
    updates = [
        make_update("client_1", 1000, 0.9, 2.0),
        make_update("client_2", 500, 0.8, 4.0),
        make_update("client_3", 500, 0.6, 6.0),
    ]

    state, weights = adaptive_averaging(updates)

    expected = (
        2.0 * 0.5625
        + 4.0 * 0.25
        + 6.0 * 0.1875
    )

    assert state["weight"].item() == pytest.approx(expected)
    assert set(weights) == {
        "client_1", "client_2", "client_3"
    }


def test_equal_accuracy_matches_fedavg():
    updates = [
        make_update("client_1", 100, 0.8, 2.0),
        make_update("client_2", 300, 0.8, 6.0),
    ]

    weights = calculate_adaptive_weights(updates)

    assert weights["client_1"] == pytest.approx(0.25)
    assert weights["client_2"] == pytest.approx(0.75)


def test_higher_accuracy_gets_higher_weight():
    updates = [
        make_update("client_1", 100, 0.9, 2.0),
        make_update("client_2", 100, 0.5, 6.0),
    ]

    weights = calculate_adaptive_weights(updates)

    assert weights["client_1"] > weights["client_2"]


def test_zero_accuracy_is_handled():
    updates = [
        make_update("client_1", 100, 0.0, 2.0),
        make_update("client_2", 100, 0.8, 6.0),
    ]

    weights = calculate_adaptive_weights(updates)

    assert weights["client_1"] > 0
    assert sum(weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("accuracy", [-0.1, 1.1, float("nan")])
def test_invalid_accuracy_rejected(accuracy):
    updates = [
        make_update("client_1", 100, accuracy, 2.0),
        make_update("client_2", 100, 0.8, 6.0),
    ]

    with pytest.raises(ValueError):
        calculate_adaptive_weights(updates)


def test_empty_updates_rejected():
    with pytest.raises(ValueError):
        adaptive_averaging([])
