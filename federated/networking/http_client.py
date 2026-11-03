import time

import httpx

from federated.networking.protocol import (
    ClientUpdateMessage,
    TrainingAssignment,
)


class FederatedHTTPClient:
    RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

    def __init__(
        self,
        client_id: str,
        server_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
    ):
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("Client ID cannot be empty")

        if timeout <= 0:
            raise ValueError("Timeout must be positive")

        if max_retries < 0:
            raise ValueError("Retries cannot be negative")

        if backoff_factor < 0:
            raise ValueError("Backoff factor cannot be negative")

        self.client_id = client_id
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def _request(
        self,
        method: str,
        path: str,
        *,
        retry: bool = True,
        **kwargs,
    ) -> dict:
        attempts = self.max_retries + 1 if retry else 1

        for attempt in range(attempts):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.request(
                        method,
                        f"{self.server_url}{path}",
                        **kwargs,
                    )

                    response.raise_for_status()
                    return response.json()

            except httpx.HTTPStatusError as exc:
                if (
                    not retry
                    or exc.response.status_code
                    not in self.RETRYABLE_STATUS_CODES
                    or attempt == attempts - 1
                ):
                    raise

            except httpx.RequestError:
                if not retry or attempt == attempts - 1:
                    raise

            delay = self.backoff_factor * (2 ** attempt)

            if delay > 0:
                time.sleep(delay)

        raise RuntimeError("HTTP request failed unexpectedly")

    def register(self) -> dict:
        return self._request(
            "POST",
            "/api/network/clients/register",
            json={"client_id": self.client_id},
            retry=False,
        )

    def heartbeat(self) -> dict:
        return self._request(
            "POST",
            f"/api/network/clients/{self.client_id}/heartbeat",
        )

    def get_assignment(
        self,
        experiment_id: int,
        round_number: int,
    ) -> TrainingAssignment:
        data = self._request(
            "GET",
            f"/api/network/clients/{self.client_id}"
            f"/assignments/{experiment_id}/{round_number}",
        )

        return TrainingAssignment.model_validate(data)

    def submit_update(
        self,
        update: ClientUpdateMessage,
    ) -> dict:
        if update.client_id != self.client_id:
            raise ValueError("Client ID mismatch")

        return self._request(
            "POST",
            f"/api/network/clients/{self.client_id}/updates",
            json=update.model_dump(),
            retry=True,
        )

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
