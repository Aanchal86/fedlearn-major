
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator  


class ExperimentConfig(BaseModel):
    """
    Validated configuration for a federated learning experiment.
    """

    model_config = ConfigDict(extra="forbid")

    dataset: Literal["fashion_mnist", "pneumonia_mnist"] = "fashion_mnist"
    model_name: Literal["small_cnn"] = "small_cnn"

    num_clients: int = Field(default=5, ge=2, le=100)
    num_rounds: int = Field(default=10, ge=1, le=1000)
    local_epochs: int = Field(default=1, ge=1, le=100)

    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)

    optimizer: Literal["adam"] = "adam"
    aggregation_method: Literal["fedavg", "adaptive"] = "fedavg"

    partition_type: Literal["iid", "dirichlet"] = "iid"
    dirichlet_alpha: float = Field(default=0.5, gt=0)

    noise_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    noisy_client_id: int = Field(default=3, ge=1)

    random_seed: int = Field(default=42, ge=0)
    device: Literal["cpu", "cuda"] = "cpu"

    @model_validator(mode="after")
    def validate_noisy_client(self):
        if self.noise_rate > 0 and self.noisy_client_id > self.num_clients:
            raise ValueError(
                "noisy_client_id cannot exceed num_clients"
            )
        return self
