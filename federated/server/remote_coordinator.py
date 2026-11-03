
import math
import time
from collections import OrderedDict

import torch

from federated.aggregation.adaptive import adaptive_averaging
from federated.aggregation.fedavg import federated_averaging
from federated.contracts.client_update import ClientUpdate
from federated.contracts.experiment import ExperimentConfig
from federated.contracts.round_result import RoundResult
from federated.networking.exchange import TrainingExchange
from federated.networking.protocol import (
    TrainingAssignment,
    decode_model_payload,
    encode_model_payload,
)
from federated.networking.registry import ClientRegistry
from federated.networking.serialization import (
    deserialize_model_state,
    serialize_model_state,
)
from federated.server.coordinator import FederatedCoordinator
from federated.training.evaluator import evaluate_model


class RemoteFederatedCoordinator:
    def __init__(
        self,
        config: ExperimentConfig,
        experiment_id: int,
        exchange: TrainingExchange,
        registry: ClientRegistry,
        timeout_seconds: float = 120.0,
        poll_interval: float = 0.5,
        min_participation_rate: float = 1.0,
    ):
        if experiment_id < 1:
            raise ValueError("Experiment ID must be positive")

        if timeout_seconds <= 0 or poll_interval <= 0:
            raise ValueError("Timeout and poll interval must be positive")

        if (
            not math.isfinite(min_participation_rate)
            or not 0 < min_participation_rate <= 1
        ):
            raise ValueError(
                "Minimum participation rate must be between 0 and 1"
            )

        self.config = config
        self.experiment_id = experiment_id
        self.exchange = exchange
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.min_participation_rate = min_participation_rate

        local = FederatedCoordinator(config)

        self.global_model = local.global_model
        self.val_loader = local.val_loader
        self.partitions = local.partitions

        self.history = []
        self.client_history = {}
        self.round_participation = {}

    def client_ids(self):
        return [
            f"client_{index + 1}"
            for index in sorted(self.partitions)
        ]

    def required_client_count(self, total_clients=None):
        if total_clients is None:
            total_clients = len(self.client_ids())

        if total_clients < 1:
            raise ValueError("At least one client is required")

        return max(
            1,
            math.ceil(
                total_clients * self.min_participation_rate
            ),
        )

    def publish_round(self, round_number: int):
        clients = self.client_ids()

        for client_id in clients:
            if self.registry.get(client_id) is None:
                raise RuntimeError(
                    f"Client not registered: {client_id}"
                )

        state = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in self.global_model.state_dict().items()
        )

        payload = encode_model_payload(
            serialize_model_state(state)
        )

        for client_id in clients:
            assignment = TrainingAssignment(
                experiment_id=self.experiment_id,
                round_number=round_number,
                dataset=self.config.dataset,
                local_epochs=self.config.local_epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                model_state_base64=payload,
            )

            self.exchange.publish_assignment(
                client_id,
                assignment,
            )

        return clients

    def validate_update(self, message, round_number):
        if message.experiment_id != self.experiment_id:
            raise ValueError("Experiment ID mismatch")

        if message.round_number != round_number:
            raise ValueError("Round number mismatch")

        clients = self.client_ids()

        if message.client_id not in clients:
            raise ValueError(
                f"Unknown client ID: {message.client_id}"
            )

        client_index = clients.index(message.client_id)
        partition_key = sorted(self.partitions)[client_index]

        expected_samples = len(
            self.partitions[partition_key]
        )

        if message.num_samples != expected_samples:
            raise ValueError(
                f"Sample count mismatch for {message.client_id}: "
                f"expected {expected_samples}, "
                f"received {message.num_samples}"
            )

        state = deserialize_model_state(
            decode_model_payload(message.model_state_base64)
        )

        reference = self.global_model.state_dict()

        if set(state.keys()) != set(reference.keys()):
            raise ValueError("Model parameter names do not match")

        for name, expected in reference.items():
            received = state[name]

            if received.shape != expected.shape:
                raise ValueError(
                    f"Model shape mismatch: {name}"
                )

            if received.dtype != expected.dtype:
                raise ValueError(
                    f"Model dtype mismatch: {name}"
                )

            if not torch.isfinite(received).all():
                raise ValueError(
                    f"Non-finite model parameter: {name}"
                )

        update = ClientUpdate(
            client_id=message.client_id,
            round_number=message.round_number,
            num_samples=message.num_samples,
            model_state=OrderedDict(
                (name, state[name])
                for name in reference
            ),
            training_loss=message.training_loss,
            training_accuracy=message.training_accuracy,
            training_time_seconds=message.training_time_seconds,
        )

        update.validate()
        return update

    def collect_updates(self, round_number, clients):
        deadline = time.monotonic() + self.timeout_seconds
        received = {}
        required = self.required_client_count(len(clients))

        while len(received) < len(clients):
            for client_id in clients:
                if client_id in received:
                    continue

                message = self.exchange.get_update(
                    client_id,
                    self.experiment_id,
                    round_number,
                )

                if message is not None:
                    if message.client_id != client_id:
                        raise ValueError("Client ID mismatch")

                    received[client_id] = self.validate_update(
                        message,
                        round_number,
                    )

            if len(received) == len(clients):
                break

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            time.sleep(min(self.poll_interval, remaining))

        missing = [
            client_id
            for client_id in clients
            if client_id not in received
        ]

        self.round_participation[round_number] = {
            "expected_clients": list(clients),
            "received_clients": list(received),
            "missing_clients": missing,
            "required_clients": required,
            "participation_rate": (
                len(received) / len(clients)
            ),
        }

        if len(received) < required:
            raise TimeoutError(
                f"Insufficient client updates: "
                f"received {len(received)}, "
                f"required {required}. "
                f"Missing client updates: {missing}"
            )

        return [
            received[client_id]
            for client_id in clients
            if client_id in received
        ]

    def aggregate_round(self, round_number, updates, round_start):
        if self.config.aggregation_method == "fedavg":
            state = federated_averaging(updates)

            total_samples = sum(
                update.num_samples
                for update in updates
            )

            weights = {
                update.client_id: (
                    update.num_samples / total_samples
                )
                for update in updates
            }

        elif self.config.aggregation_method == "adaptive":
            state, weights = adaptive_averaging(updates)

        else:
            raise ValueError("Unsupported aggregation method")

        self.global_model.load_state_dict(
            state,
            strict=True,
        )

        validation = evaluate_model(
            self.global_model,
            self.val_loader,
            device=self.config.device,
        )

        self.client_history[round_number] = [
            {
                "client_id": update.client_id,
                "num_samples": update.num_samples,
                "aggregation_weight": weights[update.client_id],
                "training_loss": update.training_loss,
                "training_accuracy": update.training_accuracy,
            }
            for update in updates
        ]

        result = RoundResult(
            round_number=round_number,
            global_accuracy=validation["accuracy"],
            global_loss=validation["loss"],
            global_macro_f1=validation["macro_f1"],
            participating_clients=[
                update.client_id
                for update in updates
            ],
            total_samples=sum(
                update.num_samples
                for update in updates
            ),
            aggregation_method=self.config.aggregation_method,
            round_time_seconds=(
                time.perf_counter() - round_start
            ),
            client_weights=weights,
        )

        self.history.append(result)
        return result

    def run_round(self, round_number: int):
        if (
            round_number < 1
            or round_number > self.config.num_rounds
        ):
            raise ValueError(
                "Round number outside configured range"
            )

        round_start = time.perf_counter()

        clients = self.publish_round(round_number)

        print(
            f"Published round {round_number} "
            f"to {len(clients)} clients"
        )

        updates = self.collect_updates(
            round_number,
            clients,
        )

        result = self.aggregate_round(
            round_number,
            updates,
            round_start,
        )

        print(
            f"Round {round_number} completed | "
            f"Clients: {len(updates)}/{len(clients)} | "
            f"Accuracy: {result.global_accuracy:.4f} | "
            f"Loss: {result.global_loss:.4f} | "
            f"Macro-F1: {result.global_macro_f1:.4f}"
        )

        return result

    def run(self):
        self.history = []
        self.client_history = {}
        self.round_participation = {}

        for round_number in range(
            1,
            self.config.num_rounds + 1,
        ):
            self.run_round(round_number)

        return self.history
