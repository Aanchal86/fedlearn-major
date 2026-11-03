
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from federated.contracts.experiment import ExperimentConfig
from federated.networking.exchange import training_exchange
from federated.networking.registry import client_registry
from federated.server.remote_coordinator import RemoteFederatedCoordinator


router = APIRouter(
    prefix="/api/network/dev/remote",
    tags=["Remote Federated Training"],
)

executor = ThreadPoolExecutor(max_workers=1)
jobs = {}
jobs_lock = Lock()


class StartRemoteTrainingRequest(BaseModel):
    experiment_id: int = Field(gt=0)
    dataset: str = "pneumonia_mnist"
    num_clients: int = Field(default=2, ge=2, le=100)
    num_rounds: int = Field(default=1, ge=1)
    local_epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    aggregation_method: str = "fedavg"
    partition_type: str = "iid"
    dirichlet_alpha: float = Field(default=0.5, gt=0)
    random_seed: int = 42
    timeout_seconds: float = Field(default=120.0, gt=0)
    min_participation_rate: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        allow_inf_nan=False,
    )


def require_dev_mode():
    if os.getenv("FEDLEARN_ENABLE_DEV_API") != "1":
        raise HTTPException(
            status_code=404,
            detail="Development API disabled",
        )


def execute_job(experiment_id, coordinator):
    try:
        results = coordinator.run()

        with jobs_lock:
            jobs[experiment_id]["status"] = "completed"
            jobs[experiment_id]["results"] = [
                result.model_dump()
                for result in results
            ]
            jobs[experiment_id]["participation"] = dict(
                coordinator.round_participation
            )

    except Exception as exc:
        with jobs_lock:
            jobs[experiment_id]["status"] = "failed"
            jobs[experiment_id]["error"] = str(exc)
            jobs[experiment_id]["participation"] = dict(
                coordinator.round_participation
            )


@router.post("/start")
def start_remote_training(payload: StartRemoteTrainingRequest):
    require_dev_mode()

    with jobs_lock:
        if payload.experiment_id in jobs:
            raise HTTPException(
                status_code=409,
                detail="Experiment job already exists",
            )

        try:
            config = ExperimentConfig(
                dataset=payload.dataset,
                num_clients=payload.num_clients,
                num_rounds=payload.num_rounds,
                local_epochs=payload.local_epochs,
                batch_size=payload.batch_size,
                learning_rate=payload.learning_rate,
                aggregation_method=payload.aggregation_method,
                partition_type=payload.partition_type,
                dirichlet_alpha=payload.dirichlet_alpha,
                random_seed=payload.random_seed,
                device="cpu",
            )

            coordinator = RemoteFederatedCoordinator(
                config=config,
                experiment_id=payload.experiment_id,
                exchange=training_exchange,
                registry=client_registry,
                timeout_seconds=payload.timeout_seconds,
                min_participation_rate=payload.min_participation_rate,
            )

        except (ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        jobs[payload.experiment_id] = {
            "status": "running",
            "results": [],
            "error": None,
            "participation": {},
            "min_participation_rate": payload.min_participation_rate,
        }

    try:
        executor.submit(
            execute_job,
            payload.experiment_id,
            coordinator,
        )
    except RuntimeError as exc:
        with jobs_lock:
            jobs.pop(payload.experiment_id, None)

        raise HTTPException(
            status_code=503,
            detail="Training executor unavailable",
        ) from exc

    return {
        "status": "started",
        "experiment_id": payload.experiment_id,
        "min_participation_rate": payload.min_participation_rate,
        "required_clients": coordinator.required_client_count(),
    }


@router.get("/status/{experiment_id}")
def get_remote_training_status(experiment_id: int):
    require_dev_mode()

    with jobs_lock:
        job = jobs.get(experiment_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Experiment job not found",
            )

        return dict(job)
