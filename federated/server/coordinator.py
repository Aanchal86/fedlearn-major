
import time

import torch
from torch.utils.data import DataLoader

from federated.aggregation.fedavg import federated_averaging
from federated.client.trainer import ClientTrainer
from federated.contracts.experiment import ExperimentConfig
from federated.contracts.round_result import RoundResult
from federated.data.dataloaders import StandardizedDataset
from federated.data.splits import get_dataset_splits
from federated.models.cnn import create_model
from federated.partitioning.iid import partition_iid
from federated.training.evaluator import evaluate_model
from federated.training.run_baseline import set_seed


class FederatedCoordinator:
    def __init__(self, config: ExperimentConfig):
        if config.partition_type != "iid":
            raise ValueError(
                "Only IID partitioning is currently supported"
            )

        if config.aggregation_method != "fedavg":
            raise ValueError(
                "Only FedAvg aggregation is currently supported"
            )

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self.config = config
        set_seed(config.random_seed)

        splits = get_dataset_splits(
            config.dataset,
            seed=config.random_seed,
        )

        train_dataset = StandardizedDataset(splits.train)
        val_dataset = StandardizedDataset(splits.val)

        self.partitions = partition_iid(
            train_dataset,
            num_clients=config.num_clients,
            seed=config.random_seed,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.global_model = create_model(
            config.dataset
        ).to(config.device)

        self.history = []
        self.client_history = {}

    def run(self) -> list[RoundResult]:
        self.history = []
        self.client_history = {}

        for round_number in range(
            1,
            self.config.num_rounds + 1,
        ):
            print(
                f"\n{'=' * 60}\n"
                f"Federated Round "
                f"{round_number}/{self.config.num_rounds}\n"
                f"{'=' * 60}"
            )

            round_start = time.perf_counter()
            updates = []

            for client_id, dataset in self.partitions.items():
                trainer = ClientTrainer(
                    client_id=f"client_{client_id + 1}",
                    dataset=dataset,
                    batch_size=self.config.batch_size,
                    seed=self.config.random_seed + client_id,
                )

                update = trainer.train(
                    global_model=self.global_model,
                    round_number=round_number,
                    local_epochs=self.config.local_epochs,
                    learning_rate=self.config.learning_rate,
                    device=self.config.device,
                )

                updates.append(update)

            aggregated_state = federated_averaging(updates)

            self.global_model.load_state_dict(
                aggregated_state
            )

            validation = evaluate_model(
                self.global_model,
                self.val_loader,
                device=self.config.device,
            )

            total_samples = sum(
                update.num_samples
                for update in updates
            )

            client_weights = {
                update.client_id: (
                    update.num_samples / total_samples
                )
                for update in updates
            }

            self.client_history[round_number] = [
                {
                    "client_id": update.client_id,
                    "num_samples": update.num_samples,
                    "aggregation_weight": client_weights[
                        update.client_id
                    ],
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
                total_samples=total_samples,
                aggregation_method="fedavg",
                round_time_seconds=(
                    time.perf_counter() - round_start
                ),
                client_weights=client_weights,
            )

            self.history.append(result)

            print(
                f"\nRound {round_number} completed | "
                f"Val Loss: {result.global_loss:.4f} | "
                f"Val Accuracy: {result.global_accuracy:.4f} | "
                f"Val Macro-F1: {result.global_macro_f1:.4f}"
            )

        return self.history
