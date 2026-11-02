
from collections import OrderedDict

import torch
from torch.utils.data import TensorDataset

from federated.models.cnn import create_model
from federated.networking.protocol import (
    TrainingAssignment,
    decode_model_payload,
    encode_model_payload,
)
from federated.networking.serialization import (
    deserialize_model_state,
    serialize_model_state,
)
from federated.networking.worker import RemoteClientWorker


def test_remote_worker_training():
    torch.manual_seed(42)

    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 2, (8,))

    dataset = TensorDataset(images, labels)

    global_model = create_model("pneumonia_mnist")

    assignment = TrainingAssignment(
        experiment_id=1,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        model_state_base64=encode_model_payload(
            serialize_model_state(
                OrderedDict(global_model.state_dict())
            )
        ),
    )

    worker = RemoteClientWorker(
        client_id="client_1",
        dataset=dataset,
    )

    result = worker.execute(assignment)

    assert result.experiment_id == 1
    assert result.client_id == "client_1"
    assert result.round_number == 1
    assert result.num_samples == 8
    assert 0 <= result.training_accuracy <= 1

    restored_state = deserialize_model_state(
        decode_model_payload(
            result.model_state_base64
        )
    )

    assert set(restored_state) == set(global_model.state_dict())


def test_remote_worker_rejects_incompatible_model():
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 2, (4,))

    dataset = TensorDataset(images, labels)

    incompatible_model = create_model("fashion_mnist")

    assignment = TrainingAssignment(
        experiment_id=1,
        round_number=1,
        dataset="pneumonia_mnist",
        local_epochs=1,
        batch_size=4,
        learning_rate=0.001,
        model_state_base64=encode_model_payload(
            serialize_model_state(
                OrderedDict(incompatible_model.state_dict())
            )
        ),
    )

    worker = RemoteClientWorker(
        client_id="client_1",
        dataset=dataset,
    )

    try:
        worker.execute(assignment)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "Incompatible model should be rejected"
        )
