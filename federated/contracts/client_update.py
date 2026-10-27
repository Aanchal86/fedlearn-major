
from dataclasses import dataclass
from collections import OrderedDict

import torch


@dataclass
class ClientUpdate:
    """
    Represents one client's local training result.

    Model parameters remain as PyTorch tensors during
    simulated federated learning.
    """

    client_id: str
    round_number: int
    num_samples: int

    model_state: OrderedDict[str, torch.Tensor]

    training_loss: float
    training_accuracy: float
    training_time_seconds: float

    def validate(self) -> None:
        """Validate client update metadata and tensors."""

        if not self.client_id.strip():
            raise ValueError("Client ID cannot be empty")

        if self.round_number < 1:
            raise ValueError("Round number must be >= 1")

        if self.num_samples <= 0:
            raise ValueError("Number of samples must be positive")

        if not 0.0 <= self.training_accuracy <= 1.0:
            raise ValueError("Training accuracy must be between 0 and 1")

        if not 0.0 <= self.training_loss < float("inf"):
            raise ValueError("Training loss must be finite and non-negative")

        if not 0.0 <= self.training_time_seconds < float("inf"):
            raise ValueError("Training time must be finite and non-negative")

        if not self.model_state:
            raise ValueError("Model state cannot be empty")

        for name, tensor in self.model_state.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a PyTorch tensor")

            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"{name} contains NaN or infinite values"
                )
