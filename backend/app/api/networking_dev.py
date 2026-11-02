
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from federated.networking.dev_publisher import (
    publish_development_assignment,
)
from federated.networking.exchange import training_exchange
from federated.networking.registry import client_registry


router = APIRouter(
    prefix="/api/network/dev",
    tags=["Networking Development"],
)


class PublishAssignmentRequest(BaseModel):
    client_id: str = Field(min_length=1)
    experiment_id: int = Field(gt=0)
    round_number: int = Field(gt=0)
    dataset: str = "pneumonia_mnist"
    local_epochs: int = Field(default=1, gt=0)
    batch_size: int = Field(default=64, gt=0)
    learning_rate: float = Field(default=0.001, gt=0)


@router.post("/assignments")
def publish_assignment(payload: PublishAssignmentRequest):
    if os.getenv("FEDLEARN_ENABLE_DEV_API") != "1":
        raise HTTPException(
            status_code=404,
            detail="Development API disabled",
        )

    if client_registry.get(payload.client_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Client is not registered",
        )

    try:
        assignment = publish_development_assignment(
            exchange=training_exchange,
            client_id=payload.client_id,
            experiment_id=payload.experiment_id,
            round_number=payload.round_number,
            dataset=payload.dataset,
            local_epochs=payload.local_epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "status": "published",
        "client_id": payload.client_id,
        "experiment_id": assignment.experiment_id,
        "round_number": assignment.round_number,
    }
