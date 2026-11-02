
import time

import httpx

from federated.networking.protocol import (
    ClientUpdateMessage,
    TrainingAssignment,
)


class FederatedHTTPClient:
    def __init__(
        self,
        client_id: str,
        server_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
    ):
        if not client_id.strip():
            raise ValueError("Client ID cannot be empty")

        self.client_id = client_id
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def register(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/api/network/clients/register",
                json={"client_id": self.client_id},
            )

            response.raise_for_status()
            return response.json()

    def heartbeat(self) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/api/network/clients/"
                f"{self.client_id}/heartbeat",
            )

            response.raise_for_status()
            return response.json()

    def get_assignment(
        self,
        experiment_id: int,
        round_number: int,
    ) -> TrainingAssignment:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.server_url}/api/network/clients/"
                f"{self.client_id}/assignments/"
                f"{experiment_id}/{round_number}",
            )

            response.raise_for_status()

            return TrainingAssignment.model_validate(
                response.json()
            )

    def submit_update(
        self,
        update: ClientUpdateMessage,
    ) -> dict:
        if update.client_id != self.client_id:
            raise ValueError("Client ID mismatch")

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.server_url}/api/network/clients/"
                f"{self.client_id}/updates",
                json=update.model_dump(),
            )

            response.raise_for_status()
            return response.json()

    def wait_for_assignment(
        self,
        experiment_id: int,
        round_number: int,
        poll_interval: float = 2.0,
        max_wait_seconds: float = 60.0,
    ) -> TrainingAssignment:
        if poll_interval <= 0 or max_wait_seconds <= 0:
            raise ValueError("Polling times must be positive")

        deadline = time.monotonic() + max_wait_seconds

        while True:
            try:
                return self.get_assignment(
                    experiment_id,
                    round_number,
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                raise TimeoutError(
                    "Training assignment was not received"
                )

            time.sleep(min(poll_interval, remaining))
