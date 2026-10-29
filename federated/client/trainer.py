
from collections import OrderedDict
import copy
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from federated.contracts.client_update import ClientUpdate
from federated.training.trainer import train_model


class ClientTrainer:
    """
    Train one federated client independently.

    Each client starts from the supplied global model
    and trains only on its own local dataset.
    """

    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        batch_size: int = 64,
        seed: int = 42,
    ):
        if not client_id.strip():
            raise ValueError("Client ID cannot be empty")

        if len(dataset) == 0:
            raise ValueError("Client dataset cannot be empty")

        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        self.client_id = client_id
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

    def train(
        self,
        global_model: nn.Module,
        round_number: int,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ) -> ClientUpdate:
        if round_number < 1:
            raise ValueError("Round number must be >= 1")

        if local_epochs < 1:
            raise ValueError("Local epochs must be positive")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        # Copy the global model to avoid modifying it.
        local_model = copy.deepcopy(global_model)

        # Use a client-specific generator for data shuffling.
        generator = torch.Generator().manual_seed(self.seed + round_number)

        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

        start_time = time.perf_counter()

        history = train_model(
            model=local_model,
            train_loader=train_loader,
            epochs=local_epochs,
            learning_rate=learning_rate,
            device=device,
        )

        training_time = time.perf_counter() - start_time

        # The final epoch provides the reported metrics.
        final_metrics = history[-1]

        model_state = OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in local_model.state_dict().items()
        )

        update = ClientUpdate(
            client_id=self.client_id,
            round_number=round_number,
            num_samples=len(self.dataset),
            model_state=model_state,
            training_loss=final_metrics["loss"],
            training_accuracy=final_metrics["accuracy"],
            training_time_seconds=training_time,
        )

        update.validate()

        return update
