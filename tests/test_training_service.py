
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.db.database import Base
from backend.app.db.models import (
    Experiment,
    TrainingRound,
    ClientParticipation,
)
from backend.app.services import training
from federated.contracts.experiment import ExperimentConfig
from federated.contracts.round_result import RoundResult


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    yield factory

    engine.dispose()


def create_experiment(session_factory, status="running"):
    with session_factory() as db:
        experiment = Experiment(
            name="Integration Test",
            status=status,
            configuration=ExperimentConfig(
                dataset="pneumonia_mnist",
                num_clients=2,
                num_rounds=1,
            ).model_dump(),
        )

        db.add(experiment)
        db.commit()
        return experiment.id


class FakeCoordinator:
    def __init__(self, config):
        self.config = config
        self.client_history = {
            1: [
                {
                    "client_id": "client_1",
                    "num_samples": 100,
                    "aggregation_weight": 0.4,
                    "training_loss": 0.45,
                    "training_accuracy": 0.85,
                },
                {
                    "client_id": "client_2",
                    "num_samples": 150,
                    "aggregation_weight": 0.6,
                    "training_loss": 0.35,
                    "training_accuracy": 0.90,
                },
            ]
        }

    def run(self):
        return [
            RoundResult(
                round_number=1,
                global_accuracy=0.88,
                global_loss=0.32,
                global_macro_f1=0.87,
                participating_clients=[
                    "client_1",
                    "client_2",
                ],
                total_samples=250,
                aggregation_method="fedavg",
                round_time_seconds=2.5,
                client_weights={
                    "client_1": 0.4,
                    "client_2": 0.6,
                },
            )
        ]


def test_training_saves_rounds_and_clients(
    monkeypatch,
    session_factory,
):
    monkeypatch.setattr(
        training,
        "FederatedCoordinator",
        FakeCoordinator,
    )

    experiment_id = create_experiment(session_factory)

    training.run_experiment_training(
        experiment_id,
        session_factory=session_factory,
    )

    with session_factory() as db:
        experiment = db.get(Experiment, experiment_id)

        assert experiment.status == "completed"

        rounds = db.scalars(
            select(TrainingRound).where(
                TrainingRound.experiment_id == experiment_id
            )
        ).all()

        assert len(rounds) == 1
        assert rounds[0].round_number == 1
        assert rounds[0].global_accuracy == pytest.approx(0.88)
        assert rounds[0].global_macro_f1 == pytest.approx(0.87)

        clients = db.scalars(
            select(ClientParticipation).where(
                ClientParticipation.round_id == rounds[0].id
            ).order_by(ClientParticipation.client_id)
        ).all()

        assert len(clients) == 2
        assert clients[0].client_id == "client_1"
        assert clients[0].num_samples == 100
        assert clients[0].aggregation_weight == pytest.approx(0.4)
        assert clients[0].training_loss == pytest.approx(0.45)
        assert clients[0].training_accuracy == pytest.approx(0.85)

        assert clients[1].client_id == "client_2"
        assert clients[1].num_samples == 150
        assert clients[1].aggregation_weight == pytest.approx(0.6)


def test_training_failure_marks_experiment_failed(
    monkeypatch,
    session_factory,
):
    class FailingCoordinator:
        def __init__(self, config):
            pass

        def run(self):
            raise RuntimeError("Simulated training failure")

    monkeypatch.setattr(
        training,
        "FederatedCoordinator",
        FailingCoordinator,
    )

    experiment_id = create_experiment(session_factory)

    training.run_experiment_training(
        experiment_id,
        session_factory=session_factory,
    )

    with session_factory() as db:
        experiment = db.get(Experiment, experiment_id)

        assert experiment.status == "failed"

        rounds = db.scalars(
            select(TrainingRound).where(
                TrainingRound.experiment_id == experiment_id
            )
        ).all()

        assert rounds == []


def test_training_does_not_restart_completed_experiment(
    monkeypatch,
    session_factory,
):
    class UnexpectedCoordinator:
        def __init__(self, config):
            raise AssertionError(
                "Coordinator should not be initialized"
            )

    monkeypatch.setattr(
        training,
        "FederatedCoordinator",
        UnexpectedCoordinator,
    )

    experiment_id = create_experiment(
        session_factory,
        status="completed",
    )

    training.run_experiment_training(
        experiment_id,
        session_factory=session_factory,
    )

    with session_factory() as db:
        experiment = db.get(Experiment, experiment_id)
        assert experiment.status == "completed"


def test_training_missing_experiment(
    monkeypatch,
    session_factory,
):
    class UnexpectedCoordinator:
        def __init__(self, config):
            raise AssertionError(
                "Coordinator should not be initialized"
            )

    monkeypatch.setattr(
        training,
        "FederatedCoordinator",
        UnexpectedCoordinator,
    )

    training.run_experiment_training(
        999,
        session_factory=session_factory,
    )

    with session_factory() as db:
        assert db.get(Experiment, 999) is None
