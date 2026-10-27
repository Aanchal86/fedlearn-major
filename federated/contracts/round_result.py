
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RoundResult(BaseModel):
    """
    Metrics and metadata for one completed
    federated learning round.
    """

    model_config = ConfigDict(extra="forbid")

    round_number: int = Field(ge=1)

    global_accuracy: float = Field(ge=0.0, le=1.0)
    global_loss: float = Field(ge=0.0, allow_inf_nan=False)
    global_macro_f1: float = Field(ge=0.0, le=1.0)

    participating_clients: list[str] = Field(min_length=1)

    total_samples: int = Field(gt=0)

    aggregation_method: str

    round_time_seconds: float = Field(
        ge=0.0,
        allow_inf_nan=False
    )

    client_weights: dict[str, float] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def validate_clients_and_weights(self):
        if len(self.participating_clients) != len(
            set(self.participating_clients)
        ):
            raise ValueError("Duplicate participating clients")

        if any(not client.strip() for client in self.participating_clients):
            raise ValueError("Client IDs cannot be empty")

        if self.client_weights:
            if set(self.client_weights) != set(self.participating_clients):
                raise ValueError(
                    "Client weights must match participating clients"
                )

            if any(
                weight < 0 or weight > 1
                for weight in self.client_weights.values()
            ):
                raise ValueError("Client weights must be between 0 and 1")

            if abs(sum(self.client_weights.values()) - 1.0) > 1e-6:
                raise ValueError("Client weights must sum to 1")

        return self
