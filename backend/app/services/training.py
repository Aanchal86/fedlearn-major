
import logging

from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import (
    Experiment,
    TrainingRound,
    ClientParticipation,
)
from federated.contracts.experiment import ExperimentConfig
from federated.server.coordinator import FederatedCoordinator


logger = logging.getLogger(__name__)


def run_experiment_training(
    experiment_id: int,
    session_factory=None,
):
    """
    Execute federated training and persist round
    metrics and client participation records.
    """

    if session_factory is None:
        session_factory = SessionLocal

    with session_factory() as db:
        experiment = db.get(Experiment, experiment_id)

        if experiment is None:
            logger.error(
                "Experiment %s was not found",
                experiment_id,
            )
            return

        if experiment.status != "running":
            logger.warning(
                "Experiment %s is not running",
                experiment_id,
            )
            return

        try:
            config = ExperimentConfig.model_validate(
                experiment.configuration
            )

            coordinator = FederatedCoordinator(config)
            history = coordinator.run()

            for result in history:
                training_round = TrainingRound(
                    experiment_id=experiment_id,
                    round_number=result.round_number,
                    global_accuracy=result.global_accuracy,
                    global_loss=result.global_loss,
                    global_macro_f1=result.global_macro_f1,
                    round_time_seconds=result.round_time_seconds,
                )

                client_metrics = coordinator.client_history.get(
                    result.round_number,
                    [],
                )

                for client in client_metrics:
                    participation = ClientParticipation(
                        client_id=client["client_id"],
                        num_samples=client["num_samples"],
                        aggregation_weight=client[
                            "aggregation_weight"
                        ],
                        training_loss=client["training_loss"],
                        training_accuracy=client[
                            "training_accuracy"
                        ],
                    )

                    training_round.clients.append(
                        participation
                    )

                db.add(training_round)

            experiment.status = "completed"
            db.commit()

            logger.info(
                "Experiment %s completed successfully",
                experiment_id,
            )

        except Exception:
            db.rollback()

            experiment = db.get(Experiment, experiment_id)

            if experiment is not None:
                experiment.status = "failed"
                db.commit()

            logger.exception(
                "Experiment %s failed",
                experiment_id,
            )
