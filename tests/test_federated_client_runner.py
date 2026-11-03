from unittest.mock import Mock

import pytest

from federated.networking.client_runner import FederatedClientRunner
from federated.networking.protocol import TrainingAssignment


@pytest.fixture
def runner():
    instance = FederatedClientRunner.__new__(
        FederatedClientRunner
    )
    instance.client = Mock()
    instance.worker = Mock()
    return instance


@pytest.fixture
def assignment():
    return TrainingAssignment(
        experiment_id=1,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        model_state_base64="bW9kZWw=",
    )


@pytest.fixture
def update():
    return Mock(
        experiment_id=1,
        round_number=1,
        client_id="client_1",
    )


def test_runner_registers_client(runner, assignment, update):
    runner.client.wait_for_assignment.return_value = assignment
    runner.worker.execute.return_value = update
    runner.client.submit_update.return_value = {
        "status": "accepted"
    }

    result = runner.run(
        experiment_id=1,
        round_number=1,
        register=True,
    )

    runner.client.register.assert_called_once()
    assert result["status"] == "accepted"


def test_runner_skips_registration(runner, assignment, update):
    runner.client.wait_for_assignment.return_value = assignment
    runner.worker.execute.return_value = update
    runner.client.submit_update.return_value = {
        "status": "accepted"
    }

    result = runner.run(
        experiment_id=1,
        round_number=1,
        register=False,
    )

    runner.client.register.assert_not_called()
    assert result["status"] == "accepted"


def test_runner_waits_for_correct_assignment(
    runner, assignment, update
):
    runner.client.wait_for_assignment.return_value = assignment
    runner.worker.execute.return_value = update

    runner.run(
        experiment_id=1,
        round_number=1,
        max_wait_seconds=30,
    )

    runner.client.wait_for_assignment.assert_called_once_with(
        experiment_id=1,
        round_number=1,
        max_wait_seconds=30,
    )


def test_runner_executes_worker(runner, assignment, update):
    runner.client.wait_for_assignment.return_value = assignment
    runner.worker.execute.return_value = update

    runner.run(
        experiment_id=1,
        round_number=1,
    )

    runner.worker.execute.assert_called_once_with(assignment)


def test_runner_submits_update(runner, assignment, update):
    runner.client.wait_for_assignment.return_value = assignment
    runner.worker.execute.return_value = update

    runner.run(
        experiment_id=1,
        round_number=1,
    )

    runner.client.submit_update.assert_called_once_with(update)


def test_runner_rejects_wrong_experiment(runner, assignment):
    wrong = assignment.model_copy(
        update={"experiment_id": 999}
    )
    runner.client.wait_for_assignment.return_value = wrong

    with pytest.raises(
        ValueError,
        match="Experiment ID mismatch",
    ):
        runner.run(experiment_id=1, round_number=1)

    runner.worker.execute.assert_not_called()


def test_runner_rejects_wrong_round(runner, assignment):
    wrong = assignment.model_copy(
        update={"round_number": 999}
    )
    runner.client.wait_for_assignment.return_value = wrong

    with pytest.raises(
        ValueError,
        match="Round number mismatch",
    ):
        runner.run(experiment_id=1, round_number=1)

    runner.worker.execute.assert_not_called()


def test_runner_rejects_invalid_experiment_id(runner):
    with pytest.raises(
        ValueError,
        match="Experiment ID must be positive",
    ):
        runner.run(experiment_id=0, round_number=1)


def test_runner_rejects_invalid_round_number(runner):
    with pytest.raises(
        ValueError,
        match="Round number must be positive",
    ):
        runner.run(experiment_id=1, round_number=0)


def test_runner_rejects_update_experiment_mismatch(
    runner, assignment, update
):
    update.experiment_id = 999
    runner.client.wait_for_assignment.return_value = assignment
    runner.worker.execute.return_value = update

    with pytest.raises(
        ValueError,
        match="Update experiment ID mismatch",
    ):
        runner.run(experiment_id=1, round_number=1)

    runner.client.submit_update.assert_not_called()


def test_runner_rejects_update_round_mismatch(
    runner, assignment, update
):
    update.round_number = 999
    runner.client.wait_for_assignment.return_value = assignment
    runner.worker.execute.return_value = update

    with pytest.raises(
        ValueError,
        match="Update round number mismatch",
    ):
        runner.run(experiment_id=1, round_number=1)

    runner.client.submit_update.assert_not_called()
