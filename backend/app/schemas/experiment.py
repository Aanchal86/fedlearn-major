
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from federated.contracts.experiment import ExperimentConfig


class ExperimentCreate(BaseModel):
    """
    Request schema for creating an experiment.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=3,
        max_length=150,
    )

    config: ExperimentConfig = Field(
        default_factory=ExperimentConfig
    )


class ExperimentResponse(BaseModel):
    """
    Response schema for a saved experiment.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    status: Literal[
        "created",
        "running",
        "completed",
        "failed",
    ]

    configuration: ExperimentConfig
    created_at: datetime


class ExperimentListResponse(BaseModel):
    """
    Response containing multiple experiments.
    """

    experiments: list[ExperimentResponse]
    total: int = Field(ge=0)
