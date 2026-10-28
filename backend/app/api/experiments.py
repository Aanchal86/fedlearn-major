
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.db.models import Experiment
from backend.app.schemas.experiment import (
    ExperimentCreate,
    ExperimentResponse,
    ExperimentListResponse,
)


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
    """
    Create a new federated learning experiment.

    This saves the configuration but does not
    start training.
    """

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
    """Return all saved experiments."""

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
    """Return a specific experiment by ID."""

    experiment = db.get(Experiment, experiment_id)

    if experiment is None:
        raise HTTPException(
            status_code=404,
            detail="Experiment not found",
        )

    return experiment
