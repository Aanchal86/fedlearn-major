
from collections import OrderedDict

import pytest
import torch

from federated.networking.serialization import (
    serialize_model_state,
    deserialize_model_state,
)


def test_model_state_round_trip():
    original = OrderedDict(
        {
            "layer.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]]
            ),
            "layer.bias": torch.tensor([0.5, -0.5]),
        }
    )

    payload = serialize_model_state(original)
    restored = deserialize_model_state(payload)

    assert list(restored.keys()) == list(original.keys())

    for name in original:
        assert torch.equal(original[name], restored[name])


def test_reject_empty_model_state():
    with pytest.raises(ValueError):
        serialize_model_state(OrderedDict())


def test_reject_non_finite_tensor():
    state = OrderedDict(
        {"weight": torch.tensor([float("nan")])}
    )

    with pytest.raises(ValueError):
        serialize_model_state(state)


def test_reject_invalid_payload():
    with pytest.raises(ValueError):
        deserialize_model_state(b"invalid data")


def test_reject_empty_payload():
    with pytest.raises(ValueError):
        deserialize_model_state(b"")


def test_reject_oversized_payload():
    from federated.networking.serialization import (
        MAX_PAYLOAD_BYTES,
    )

    with pytest.raises(ValueError):
        deserialize_model_state(
            b"x" * (MAX_PAYLOAD_BYTES + 1)
        )
