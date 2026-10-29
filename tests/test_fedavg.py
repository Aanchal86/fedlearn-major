
from collections import OrderedDict
import copy

import pytest
import torch

from federated.aggregation.fedavg import federated_averaging
from federated.contracts.client_update import ClientUpdate
from federated.models.cnn import SmallCNN


def make_update(client_id, samples, value, round_number=1):
    return ClientUpdate(
        client_id=client_id,
        round_number=round_number,
        num_samples=samples,
        model_state=OrderedDict({
            "weight": torch.tensor([value], dtype=torch.float32)
        }),
        training_loss=0.5,
        training_accuracy=0.8,
        training_time_seconds=1.0,
    )


def test_weighted_average():
    updates = [
        make_update("client_1", 100, 0.2),
        make_update("client_2", 150, 0.4),
        make_update("client_3", 250, 0.8),
    ]

    result = federated_averaging(updates)

    assert torch.allclose(
        result["weight"],
        torch.tensor([0.56]),
        atol=1e-6,
    )


def test_equal_client_sizes():
    updates = [
        make_update("client_1", 100, 2.0),
        make_update("client_2", 100, 4.0),
        make_update("client_3", 100, 6.0),
    ]

    result = federated_averaging(updates)

    assert torch.allclose(
        result["weight"],
        torch.tensor([4.0]),
    )


def test_original_updates_unchanged():
    updates = [
        make_update("client_1", 100, 2.0),
        make_update("client_2", 200, 4.0),
    ]

    original = [
        copy.deepcopy(update.model_state)
        for update in updates
    ]

    result = federated_averaging(updates)

    for update, original_state in zip(updates, original):
        assert torch.equal(
            update.model_state["weight"],
            original_state["weight"],
        )

    result["weight"].fill_(99.0)

    for update, original_state in zip(updates, original):
        assert torch.equal(
            update.model_state["weight"],
            original_state["weight"],
        )


def test_invalid_client_updates():
    with pytest.raises(ValueError):
        federated_averaging([])

    with pytest.raises(ValueError):
        federated_averaging([
            make_update("client_1", 100, 1.0),
            make_update("client_1", 200, 2.0),
        ])

    with pytest.raises(ValueError):
        federated_averaging([
            make_update("client_1", 100, 1.0, round_number=1),
            make_update("client_2", 100, 2.0, round_number=2),
        ])

    with pytest.raises(ValueError):
        federated_averaging([
            make_update("client_1", 0, 1.0),
        ])


def test_incompatible_model_states():
    first = make_update("client_1", 100, 1.0)
    second = make_update("client_2", 100, 2.0)

    second.model_state["different_name"] = (
        second.model_state.pop("weight")
    )

    with pytest.raises(ValueError):
        federated_averaging([first, second])

    second = make_update("client_2", 100, 2.0)
    second.model_state["weight"] = torch.ones(2)

    with pytest.raises(ValueError):
        federated_averaging([first, second])

    second = make_update("client_2", 100, 2.0)
    second.model_state["weight"] = torch.ones(
        1, dtype=torch.float64
    )

    with pytest.raises(ValueError):
        federated_averaging([first, second])


def test_fedavg_with_real_cnn():
    model = SmallCNN(num_classes=2)

    state_1 = copy.deepcopy(model.state_dict())
    state_2 = copy.deepcopy(model.state_dict())

    for name in state_1:
        state_1[name] = state_1[name] + 1.0
        state_2[name] = state_2[name] + 3.0

    updates = [
        ClientUpdate(
            client_id="client_1",
            round_number=1,
            num_samples=100,
            model_state=state_1,
            training_loss=0.5,
            training_accuracy=0.8,
            training_time_seconds=1.0,
        ),
        ClientUpdate(
            client_id="client_2",
            round_number=1,
            num_samples=300,
            model_state=state_2,
            training_loss=0.4,
            training_accuracy=0.9,
            training_time_seconds=1.0,
        ),
    ]

    aggregated = federated_averaging(updates)

    new_model = SmallCNN(num_classes=2)
    new_model.load_state_dict(aggregated)

    for name in aggregated:
        expected = (
            state_1[name] * 0.25 +
            state_2[name] * 0.75
        )
        assert torch.allclose(
            aggregated[name],
            expected,
            atol=1e-5,
        )
