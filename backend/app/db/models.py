
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(150), nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(30), default="created", nullable=False
    )

    configuration: Mapped[dict] = mapped_column(
        JSON, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now
    )

    rounds: Mapped[list["TrainingRound"]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan"
    )


class TrainingRound(Base):
    __tablename__ = "training_rounds"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"),
        nullable=False
    )

    round_number: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    global_accuracy: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    global_loss: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    global_macro_f1: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    round_time_seconds: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    experiment: Mapped["Experiment"] = relationship(
        back_populates="rounds"
    )

    clients: Mapped[list["ClientParticipation"]] = relationship(
        back_populates="round",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "round_number",
            name="uq_experiment_round"
        ),
    )


class ClientParticipation(Base):
    __tablename__ = "client_participation"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True
    )

    round_id: Mapped[int] = mapped_column(
        ForeignKey("training_rounds.id"),
        nullable=False
    )

    client_id: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    num_samples: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    aggregation_weight: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    training_loss: Mapped[float] = mapped_column(
        Float, nullable=True
    )

    training_accuracy: Mapped[float] = mapped_column(
        Float, nullable=True
    )

    round: Mapped["TrainingRound"] = relationship(
        back_populates="clients"
    )

    __table_args__ = (
        UniqueConstraint(
            "round_id",
            "client_id",
            name="uq_round_client"
        ),
    )
