
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    status,
)
from sqlalchemy import select, func, update
from sqlalchemy.orm import Session, selectinload

from backend.app.db.database import get_db
from backend.app.db.models import Experiment, TrainingRound
from backend.app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
)
from backend.app.services.training import run_experiment_training
from federated.contracts.experiment import ExperimentConfig


router = APIRouter(
    prefix="/api/experiments",
    tags=["Experiments"],
)


@router.post(
    "",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_experiment(
    payload: ExperimentCreate,
    db: Session = Depends(get_db),
):
    experiment = Experiment(
        name=payload.name,
        status="created",
        configuration=payload.config.model_dump(),
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    return experiment


@router.get(
    "",
    response_model=ExperimentListResponse,
)
def list_experiments(
    db: Session = Depends(get_db),
):
    experiments = db.scalars(
        select(Experiment).order_by(
            Experiment.created_at.desc(),
            Experiment.id.desc(),
        )
    ).all()

    total = db.scalar(
        select(func.count()).select_from(Experiment)
    )

    return ExperimentListResponse(
        experiments=experiments,
        total=total,
    )


@router.get(
    "/{experiment_id}",
    response_model=ExperimentResponse,
)
def get_experiment(
    experiment_id: int,
    db: Session = Depends(get_db),
):
    experiment = db.get(Experiment, experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    return experiment


@router.post(
    "/{experiment_id}/start",
    status_code=status.HTTP_202_ACCEPTED,
)
def start_experiment(
    experiment_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    experiment = db.get(Experiment, experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    config = ExperimentConfig.model_validate(
        experiment.configuration
    )

    if config.partition_type != "iid":
        raise HTTPException(
            status_code=400,
            detail="Only IID partitioning is currently supported",
        )

    if config.aggregation_method != "fedavg":
        raise HTTPException(
            status_code=400,
            detail="Only FedAvg aggregation is currently supported",
        )

    statement = (
        update(Experiment)
        .where(
            Experiment.id == experiment_id,
            Experiment.status == "created",
        )
        .values(status="running")
    )

    result = db.execute(statement)

    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Experiment has already been started",
        )

    db.commit()

    background_tasks.add_task(
        run_experiment_training,
        experiment_id,
    )

    return {
        "experiment_id": experiment_id,
        "status": "running",
        "message": "Federated training has been scheduled",
    }


@router.get("/{experiment_id}/results")
def get_experiment_results(
    experiment_id: int,
    db: Session = Depends(get_db),
):
    experiment = db.get(Experiment, experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    rounds = db.scalars(
        select(TrainingRound)
        .where(TrainingRound.experiment_id == experiment_id)
        .options(selectinload(TrainingRound.clients))
        .order_by(TrainingRound.round_number)
    ).all()

    history = []

    for training_round in rounds:
        clients = [
            {
                "client_id": client.client_id,
                "num_samples": client.num_samples,
                "aggregation_weight": client.aggregation_weight,
                "training_loss": client.training_loss,
                "training_accuracy": client.training_accuracy,
            }
            for client in sorted(
                training_round.clients,
                key=lambda item: item.client_id,
            )
        ]

        history.append(
            {
                "round_number": training_round.round_number,
                "global_accuracy": training_round.global_accuracy,
                "global_loss": training_round.global_loss,
                "global_macro_f1": training_round.global_macro_f1,
                "round_time_seconds": training_round.round_time_seconds,
                "clients": clients,
            }
        )

    return {
        "experiment_id": experiment.id,
        "name": experiment.name,
        "status": experiment.status,
        "configuration": experiment.configuration,
        "total_rounds_completed": len(history),
        "history": history,
    }

