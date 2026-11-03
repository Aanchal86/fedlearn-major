
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock


@dataclass(frozen=True)
class RegisteredClient:
    client_id: str
    registered_at: datetime
    last_seen: datetime


class ClientRegistry:
    def __init__(self):
        self._clients: dict[str, RegisteredClient] = {}
        self._lock = Lock()

    def register(self, client_id: str) -> RegisteredClient:
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("Client ID cannot be empty")

        client_id = client_id.strip()

        with self._lock:
            if client_id in self._clients:
                raise ValueError("Client is already registered")

            now = datetime.now(timezone.utc)

            client = RegisteredClient(
                client_id=client_id,
                registered_at=now,
                last_seen=now,
            )

            self._clients[client_id] = client
            return client

    def get(self, client_id: str) -> RegisteredClient | None:
        with self._lock:
            return self._clients.get(client_id)

    def heartbeat(self, client_id: str) -> RegisteredClient:
        with self._lock:
            existing = self._clients.get(client_id)

            if existing is None:
                raise KeyError("Client is not registered")

            updated = RegisteredClient(
                client_id=existing.client_id,
                registered_at=existing.registered_at,
                last_seen=datetime.now(timezone.utc),
            )

            self._clients[client_id] = updated
            return updated

    def list_clients(self) -> list[RegisteredClient]:
        with self._lock:
            return sorted(
                self._clients.values(),
                key=lambda client: client.client_id,
            )

    def is_active(
        self,
        client_id: str,
        timeout_seconds: float = 60.0,
    ) -> bool:
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        client = self.get(client_id)

        if client is None:
            return False

        now = datetime.now(timezone.utc)

        elapsed = (
            now - client.last_seen
        ).total_seconds()

        return elapsed <= timeout_seconds

    def list_active_clients(
        self,
        timeout_seconds: float = 60.0,
    ) -> list[RegisteredClient]:
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        return [
            client
            for client in self.list_clients()
            if self.is_active(
                client.client_id,
                timeout_seconds,
            )
        ]

    def list_inactive_clients(
        self,
        timeout_seconds: float = 60.0,
    ) -> list[RegisteredClient]:
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        return [
            client
            for client in self.list_clients()
            if not self.is_active(
                client.client_id,
                timeout_seconds,
            )
        ]

    def health_summary(
        self,
        timeout_seconds: float = 60.0,
    ) -> dict:
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")

        clients = self.list_clients()

        active = [
            client.client_id
            for client in clients
            if self.is_active(
                client.client_id,
                timeout_seconds,
            )
        ]

        inactive = [
            client.client_id
            for client in clients
            if client.client_id not in active
        ]

        return {
            "total_clients": len(clients),
            "active_clients": len(active),
            "inactive_clients": len(inactive),
            "active_client_ids": active,
            "inactive_client_ids": inactive,
            "timeout_seconds": timeout_seconds,
        }


client_registry = ClientRegistry()

