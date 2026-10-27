
import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.app.db.database import Base
from backend.app.db.models import (
    Experiment,
    TrainingRound,
    ClientParticipation,
)
from federated.contracts.experiment import ExperimentConfig


@pytest.fixture
def db_session():
    # Isolated in-memory SQLite database for testing.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        yield session

    engine.dispose()


def test_database_tables(db_session):
    tables = inspect(db_session.bind).get_table_names()

    assert "experiments" in tables
    assert "training_rounds" in tables
    assert "client_participation" in tables


def test_create_experiment(db_session):
    config = ExperimentConfig()

    experiment = Experiment(
        name="FedAvg Baseline",
        status="created",
        configuration=config.model_dump(),
    )

    db_session.add(experiment)
    db_session.commit()

    saved = db_session.scalar(
        select(Experiment).where(
            Experiment.name == "FedAvg Baseline"
        )
    )

    assert saved is not None
    assert saved.id is not None
    assert saved.configuration["num_clients"] == 5


def test_experiment_round_client_relationship(db_session):
    experiment = Experiment(
        name="Relationship Test",
        configuration=ExperimentConfig().model_dump(),
    )

    training_round = TrainingRound(
        round_number=1,
        global_accuracy=0.82,
        global_loss=0.48,
        global_macro_f1=0.81,
        round_time_seconds=12.5,
    )

    client = ClientParticipation(
        client_id="client_1",
        num_samples=12000,
        aggregation_weight=1.0,
        training_loss=0.45,
        training_accuracy=0.85,
    )

    training_round.clients.append(client)
    experiment.rounds.append(training_round)

    db_session.add(experiment)
    db_session.commit()

    saved = db_session.scalar(
        select(Experiment).where(
            Experiment.name == "Relationship Test"
        )
    )

    assert len(saved.rounds) == 1
    assert saved.rounds[0].round_number == 1
    assert len(saved.rounds[0].clients) == 1
    assert saved.rounds[0].clients[0].client_id == "client_1"


def test_cascade_delete(db_session):
    experiment = Experiment(
        name="Delete Test",
        configuration=ExperimentConfig().model_dump(),
    )

    training_round = TrainingRound(
        round_number=1,
        global_accuracy=0.75,
        global_loss=0.65,
        global_macro_f1=0.73,
        round_time_seconds=10.0,
    )

    client = ClientParticipation(
        client_id="client_1",
        num_samples=100,
        aggregation_weight=1.0,
    )

    training_round.clients.append(client)
    experiment.rounds.append(training_round)

    db_session.add(experiment)
    db_session.commit()

    db_session.delete(experiment)
    db_session.commit()

    assert db_session.scalar(select(Experiment)) is None
    assert db_session.scalar(select(TrainingRound)) is None
    assert db_session.scalar(select(ClientParticipation)) is None
