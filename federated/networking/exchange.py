from threading import Lock

from federated.networking.protocol import (
    ClientUpdateMessage,
    TrainingAssignment,
)


class TrainingExchange:
    def __init__(self):
        self._assignments = {}
        self._updates = {}
        self._lock = Lock()

    def publish_assignment(
        self,
        client_id: str,
        assignment: TrainingAssignment,
    ):
        if not client_id:
            raise ValueError("Client ID is required")

        key = (
            assignment.experiment_id,
            assignment.round_number,
            client_id,
        )

        with self._lock:
            if key in self._assignments:
                raise ValueError("Assignment already exists")

            self._assignments[key] = assignment

    def get_assignment(
        self,
        client_id: str,
        experiment_id: int,
        round_number: int,
    ):
        key = (experiment_id, round_number, client_id)

        with self._lock:
            return self._assignments.get(key)

    def submit_update(self, update: ClientUpdateMessage):
        key = (
            update.experiment_id,
            update.round_number,
            update.client_id,
        )

        with self._lock:
            if key not in self._assignments:
                raise KeyError("No matching training assignment")

            existing = self._updates.get(key)

            if existing is not None:
                if existing == update:
                    return False

                raise ValueError(
                    "Conflicting update already submitted"
                )

            self._updates[key] = update
            return True

    def get_update(
        self,
        client_id: str,
        experiment_id: int,
        round_number: int,
    ):
        key = (experiment_id, round_number, client_id)

        with self._lock:
            return self._updates.get(key)


training_exchange = TrainingExchange()
