
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import TensorDataset

from federated.networking.client_runner import (
    FederatedClientRunner,
)
from federated.networking.protocol import (
    ClientUpdateMessage,
    TrainingAssignment,
    encode_model_payload,
)


@pytest.fixture
def runner():
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 2, (8,))
    dataset = TensorDataset(images, labels)

    return FederatedClientRunner(
        client_id="client_1",
        dataset=dataset,
    )


@pytest.fixture
def assignment():
    return TrainingAssignment(
        experiment_id=1,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        model_state_base64=encode_model_payload(b"model"),
    )


def test_runner_registers_and_submits(
    runner,
    assignment,
    monkeypatch,
):
    register = Mock(return_value={"status": "registered"})
    wait = Mock(return_value=assignment)
    submit = Mock(return_value={"status": "accepted"})

    update = ClientUpdateMessage(
        experiment_id=1,
        client_id="client_1",
        round_number=1,
        num_samples=8,
        training_loss=0.2,
        training_accuracy=0.8,
        training_time_seconds=1.0,
        model_state_base64=encode_model_payload(b"updated-model"),
    )

    execute = Mock(return_value=update)

    monkeypatch.setattr(runner.client, "register", register)
    monkeypatch.setattr(
        runner.client,
        "wait_for_assignment",
        wait,
    )
    monkeypatch.setattr(
        runner.client,
        "submit_update",
        submit,
    )
    monkeypatch.setattr(runner.worker, "execute", execute)

    result = runner.run(
        experiment_id=1,
        round_number=1,
    )

    register.assert_called_once()
    execute.assert_called_once_with(assignment)
    submit.assert_called_once_with(update)

    assert result["status"] == "accepted"


def test_runner_skips_registration(
    runner,
    assignment,
    monkeypatch,
):
    register = Mock()
    wait = Mock(return_value=assignment)
    submit = Mock(return_value={"status": "accepted"})

    update = Mock()

    monkeypatch.setattr(runner.client, "register", register)
    monkeypatch.setattr(
        runner.client,
        "wait_for_assignment",
        wait,
    )
    monkeypatch.setattr(
        runner.client,
        "submit_update",
        submit,
    )
    monkeypatch.setattr(
        runner.worker,
        "execute",
        Mock(return_value=update),
    )

    result = runner.run(
        experiment_id=1,
        round_number=1,
        register=False,
    )

    register.assert_not_called()
    assert result["status"] == "accepted"


def test_runner_rejects_wrong_experiment(
    runner,
    assignment,
    monkeypatch,
):
    wrong = assignment.model_copy(
        update={"experiment_id": 99}
    )

    monkeypatch.setattr(
        runner.client,
        "register",
        Mock(),
    )
    monkeypatch.setattr(
        runner.client,
        "wait_for_assignment",
        Mock(return_value=wrong),
    )

    with pytest.raises(ValueError, match="Experiment ID mismatch"):
        runner.run(1, 1)


def test_runner_rejects_wrong_round(
    runner,
    assignment,
    monkeypatch,
):
    wrong = assignment.model_copy(
        update={"round_number": 99}
    )

    monkeypatch.setattr(
        runner.client,
        "register",
        Mock(),
    )
    monkeypatch.setattr(
        runner.client,
        "wait_for_assignment",
        Mock(return_value=wrong),
    )

    with pytest.raises(ValueError, match="Round number mismatch"):
        runner.run(1, 1)
