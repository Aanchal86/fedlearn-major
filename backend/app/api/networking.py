from fastapi import APIRouter, HTTPException, status

from federated.networking.protocol import (
    ClientRegistration,
    ClientRegistrationResponse,
    ClientUpdateMessage,
    TrainingAssignment,
)
from federated.networking.registry import client_registry
from federated.networking.exchange import training_exchange


router = APIRouter(
    prefix="/api/network",
    tags=["Networking"],
)


@router.post(
    "/clients/register",
    response_model=ClientRegistrationResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_client(payload: ClientRegistration):
    try:
        client_registry.register(payload.client_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return ClientRegistrationResponse(
        client_id=payload.client_id,
        status="registered",
    )


@router.get("/clients")
def list_clients():
    clients = client_registry.list_clients()

    return {
        "total": len(clients),
        "clients": [
            {
                "client_id": client.client_id,
                "registered_at": client.registered_at.isoformat(),
                "last_seen": client.last_seen.isoformat(),
            }
            for client in clients
        ],
    }


@router.post("/clients/{client_id}/heartbeat")
def client_heartbeat(client_id: str):
    try:
        client = client_registry.heartbeat(client_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Client is not registered",
        ) from exc

    return {
        "client_id": client.client_id,
        "status": "active",
        "last_seen": client.last_seen.isoformat(),
    }


@router.get(
    "/clients/{client_id}/assignments/{experiment_id}/{round_number}",
    response_model=TrainingAssignment,
)
def get_training_assignment(
    client_id: str,
    experiment_id: int,
    round_number: int,
):
    if client_registry.get(client_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Client is not registered",
        )

    assignment = training_exchange.get_assignment(
        client_id=client_id,
        experiment_id=experiment_id,
        round_number=round_number,
    )

    if assignment is None:
        raise HTTPException(
            status_code=404,
            detail="Training assignment not found",
        )

    return assignment


@router.post("/clients/{client_id}/updates")
def submit_client_update(
    client_id: str,
    update: ClientUpdateMessage,
):
    if client_registry.get(client_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Client is not registered",
        )

    if update.client_id != client_id:
        raise HTTPException(
            status_code=400,
            detail="Client ID mismatch",
        )

    try:
        inserted = training_exchange.submit_update(update)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    return {
        "status": "accepted" if inserted else "already_accepted",
        "client_id": update.client_id,
        "experiment_id": update.experiment_id,
        "round_number": update.round_number,
    }
