
from collections import OrderedDict

import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from federated.models.cnn import create_model
from federated.networking.exchange import training_exchange
from federated.networking.protocol import (
    TrainingAssignment,
    ClientUpdateMessage,
    encode_model_payload,
    decode_model_payload,
)
from federated.networking.serialization import (
    serialize_model_state,
    deserialize_model_state,
)


def test_submitted_model_update_is_valid():
    model = create_model("pneumonia_mnist")

    initial_state = OrderedDict(
        (key, value.detach().cpu().clone())
        for key, value in model.state_dict().items()
    )

    assignment = TrainingAssignment(
        experiment_id=987654,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=64,
        learning_rate=0.001,
        model_state_base64=encode_model_payload(
            serialize_model_state(initial_state)
        ),
    )

    client_id = "verification_client_987654"

    training_exchange.publish_assignment(
        client_id,
        assignment,
    )

    trained_state = OrderedDict(
        (key, value.clone())
        for key, value in initial_state.items()
    )

    first_weight = next(
        key for key, value in trained_state.items()
        if value.is_floating_point()
    )

    trained_state[first_weight] += 0.01

    update = ClientUpdateMessage(
        experiment_id=987654,
        client_id=client_id,
        round_number=1,
        num_samples=942,
        training_loss=0.5458,
        training_accuracy=0.7197,
        training_time_seconds=0.14,
        model_state_base64=encode_model_payload(
            serialize_model_state(trained_state)
        ),
    )

    training_exchange.submit_update(update)

    received = training_exchange.get_update(
        client_id,
        experiment_id=987654,
        round_number=1,
    )

    assert received is not None
    assert received.client_id == client_id
    assert received.num_samples == 942

    received_state = deserialize_model_state(
        decode_model_payload(
            received.model_state_base64
        )
    )

    verified_model = create_model("pneumonia_mnist")
    verified_model.load_state_dict(
        received_state,
        strict=True,
    )

    assert len(received_state) == len(initial_state)

    assert any(
        not torch.equal(
            initial_state[key],
            received_state[key],
        )
        for key in initial_state
    )

    for tensor in received_state.values():
        assert torch.isfinite(tensor).all()
