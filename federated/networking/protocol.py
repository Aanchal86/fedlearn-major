
import base64
import binascii
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from federated.networking.serialization import MAX_PAYLOAD_BYTES


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClientRegistration(ProtocolModel):
    client_id: str = Field(min_length=1, max_length=100)


class ClientRegistrationResponse(ProtocolModel):
    client_id: str
    status: Literal["registered"]


class TrainingAssignment(ProtocolModel):
    experiment_id: int = Field(gt=0)
    round_number: int = Field(gt=0)
    dataset: Literal["fashion_mnist", "pneumonia_mnist"]
    local_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    model_state_base64: str = Field(min_length=1)

    @field_validator("model_state_base64")
    @classmethod
    def validate_model_payload(cls, value: str) -> str:
        decode_model_payload(value)
        return value


class ClientUpdateMessage(ProtocolModel):
    experiment_id: int = Field(gt=0)
    client_id: str = Field(min_length=1, max_length=100)
    round_number: int = Field(gt=0)
    num_samples: int = Field(gt=0)
    training_loss: float = Field(ge=0, allow_inf_nan=False)
    training_accuracy: float = Field(
        ge=0, le=1, allow_inf_nan=False
    )
    training_time_seconds: float = Field(
        ge=0, allow_inf_nan=False
    )
    model_state_base64: str = Field(min_length=1)

    @field_validator("model_state_base64")
    @classmethod
    def validate_model_payload(cls, value: str) -> str:
        decode_model_payload(value)
        return value


def encode_model_payload(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("Model payload must be bytes")

    if not payload or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Invalid model payload size")

    return base64.b64encode(payload).decode("ascii")


def decode_model_payload(value: str) -> bytes:
    if not isinstance(value, str):
        raise TypeError("Encoded model payload must be a string")

    if len(value) > ((MAX_PAYLOAD_BYTES + 2) // 3) * 4:
        raise ValueError("Encoded model payload exceeds size limit")

    try:
        payload = base64.b64decode(
            value,
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid Base64 model payload") from exc

    if not payload or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("Invalid model payload size")

    return payload
