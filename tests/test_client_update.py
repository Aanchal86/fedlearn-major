
from collections import OrderedDict

import pytest
import torch

from federated.contracts.client_update import ClientUpdate


def make_update(num_samples=100):
    return ClientUpdate(
        client_id="client_1",
        round_number=1,
        num_samples=num_samples,
        model_state=OrderedDict({
            "layer.weight": torch.tensor([0.2, 0.4]),
            "layer.bias": torch.tensor([0.1]),
        }),
        training_loss=0.45,
        training_accuracy=0.85,
        training_time_seconds=2.5,
    )


def test_valid_client_update():
    update = make_update()
    update.validate()


def test_invalid_sample_count():
    update = make_update(num_samples=0)

    with pytest.raises(ValueError):
        update.validate()


def test_invalid_accuracy():
    update = make_update()
    update.training_accuracy = 1.5

    with pytest.raises(ValueError):
        update.validate()


def test_invalid_tensor():
    update = make_update()
    update.model_state["layer.weight"] = torch.tensor(
        [float("nan")]
    )

    with pytest.raises(ValueError):
        update.validate()
