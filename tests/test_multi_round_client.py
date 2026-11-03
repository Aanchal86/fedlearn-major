
from unittest.mock import MagicMock, patch

import pytest

from federated.networking.client_runner import FederatedClientRunner


@pytest.fixture
def runner():
    with (
        patch(
            "federated.networking.client_runner.FederatedHTTPClient"
        ) as mock_http,
        patch(
            "federated.networking.client_runner.RemoteClientWorker"
        ) as mock_worker,
    ):
        client = mock_http.return_value
        worker = mock_worker.return_value

        client.client_id = "client_1"

        def assignment_for_round(**kwargs):
            assignment = MagicMock()
            assignment.experiment_id = kwargs["experiment_id"]
            assignment.round_number = kwargs["round_number"]
            return assignment

        def update_for_assignment(assignment):
            update = MagicMock()
            update.experiment_id = assignment.experiment_id
            update.round_number = assignment.round_number
            return update

        client.wait_for_assignment.side_effect = (
            assignment_for_round
        )

        worker.execute.side_effect = (
            update_for_assignment
        )

        client.submit_update.side_effect = (
            lambda update: {
                "status": "accepted",
                "round_number": update.round_number,
            }
        )

        instance = FederatedClientRunner(
            client_id="client_1",
            dataset=MagicMock(),
        )

        yield instance


def test_single_round_still_works(runner):
    result = runner.run(
        experiment_id=20,
        round_number=1,
    )

    assert result["status"] == "accepted"
    runner.client.register.assert_called_once()


def test_three_rounds_execute_in_order(runner):
    results = runner.run_rounds(
        experiment_id=20,
        num_rounds=3,
    )

    assert [
        item["round_number"]
        for item in results
    ] == [1, 2, 3]

    assert [
        item["response"]["round_number"]
        for item in results
    ] == [1, 2, 3]


def test_registration_happens_once(runner):
    runner.run_rounds(
        experiment_id=20,
        num_rounds=3,
    )

    runner.client.register.assert_called_once()


def test_resume_from_later_round(runner):
    results = runner.run_rounds(
        experiment_id=20,
        num_rounds=2,
        start_round=3,
    )

    assert [
        item["round_number"]
        for item in results
    ] == [3, 4]


def test_skip_registration(runner):
    runner.run_rounds(
        experiment_id=20,
        num_rounds=2,
        register=False,
    )

    runner.client.register.assert_not_called()


def test_invalid_round_count(runner):
    with pytest.raises(ValueError):
        runner.run_rounds(
            experiment_id=20,
            num_rounds=0,
        )


def test_invalid_start_round(runner):
    with pytest.raises(ValueError):
        runner.run_rounds(
            experiment_id=20,
            num_rounds=2,
            start_round=0,
        )


def test_invalid_experiment_id(runner):
    with pytest.raises(ValueError):
        runner.run_rounds(
            experiment_id=0,
            num_rounds=2,
        )


def test_invalid_max_wait(runner):
    with pytest.raises(ValueError):
        runner.run_rounds(
            experiment_id=20,
            num_rounds=2,
            max_wait_seconds=0,
        )


def test_assignment_mismatch_rejected(runner):
    runner.client.wait_for_assignment.return_value = None
    runner.client.wait_for_assignment.side_effect = None

    wrong_assignment = MagicMock()
    wrong_assignment.experiment_id = 999
    wrong_assignment.round_number = 1

    runner.client.wait_for_assignment.return_value = (
        wrong_assignment
    )

    with pytest.raises(ValueError, match="Experiment ID mismatch"):
        runner.run(
            experiment_id=20,
            round_number=1,
        )


def test_update_round_mismatch_rejected(runner):
    wrong_update = MagicMock()
    wrong_update.experiment_id = 20
    wrong_update.round_number = 999

    runner.worker.execute.side_effect = None
    runner.worker.execute.return_value = wrong_update

    with pytest.raises(
        ValueError,
        match="Update round number mismatch",
    ):
        runner.run(
            experiment_id=20,
            round_number=1,
        )


def test_no_submission_after_failed_round(runner):
    runner.client.wait_for_assignment.side_effect = (
        lambda **kwargs: (
            (_ for _ in ()).throw(
                TimeoutError("Assignment missing")
            )
            if kwargs["round_number"] == 2
            else MagicMock(
                experiment_id=20,
                round_number=1,
            )
        )
    )

    with pytest.raises(TimeoutError):
        runner.run_rounds(
            experiment_id=20,
            num_rounds=3,
        )

    assert runner.client.submit_update.call_count == 1
