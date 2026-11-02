
import base64

import pytest
from pydantic import ValidationError

from federated.networking.protocol import (
    ClientRegistration,
    ClientUpdateMessage,
    TrainingAssignment,
    decode_model_payload,
    encode_model_payload,
)


def test_client_registration():
    registration = ClientRegistration(client_id="client_1")
    assert registration.client_id == "client_1"


def test_empty_client_id_rejected():
    with pytest.raises(ValidationError):
        ClientRegistration(client_id="")


def test_model_payload_round_trip():
    original = b"sample-model-weights"
    encoded = encode_model_payload(original)
    decoded = decode_model_payload(encoded)

    assert decoded == original


def test_invalid_base64_rejected():
    with pytest.raises(ValueError):
        decode_model_payload("invalid!!!")


def test_training_assignment():
    assignment = TrainingAssignment(
        experiment_id=1,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=64,
        learning_rate=0.001,
        model_state_base64=encode_model_payload(b"model"),
    )

    assert assignment.round_number == 1
    assert assignment.dataset == "pneumonia_mnist"


def test_client_update_message():
    update = ClientUpdateMessage(
        experiment_id=1,
        client_id="client_1",
        round_number=1,
        num_samples=100,
        training_loss=0.4,
        training_accuracy=0.85,
        training_time_seconds=2.0,
        model_state_base64=encode_model_payload(b"model"),
    )

    assert update.client_id == "client_1"
    assert update.training_accuracy == 0.85


def test_invalid_training_accuracy_rejected():
    with pytest.raises(ValidationError):
        ClientUpdateMessage(
            experiment_id=1,
            client_id="client_1",
            round_number=1,
            num_samples=100,
            training_loss=0.4,
            training_accuracy=1.5,
            training_time_seconds=2.0,
            model_state_base64=encode_model_payload(b"model"),
        )


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        ClientRegistration(
            client_id="client_1",
            secret_token="unexpected",
        )


def test_oversized_encoded_payload_rejected(monkeypatch):
    from federated.networking import protocol

    monkeypatch.setattr(
        protocol,
        "MAX_PAYLOAD_BYTES",
        8,
    )

    encoded = base64.b64encode(b"x" * 9).decode("ascii")

    with pytest.raises(ValueError):
        decode_model_payload(encoded)
