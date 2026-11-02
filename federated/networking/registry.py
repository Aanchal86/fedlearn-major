
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


client_registry = ClientRegistry()
