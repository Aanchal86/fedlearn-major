
from collections import OrderedDict

from federated.models.cnn import create_model
from federated.networking.exchange import TrainingExchange
from federated.networking.protocol import (
    TrainingAssignment,
    encode_model_payload,
)
from federated.networking.serialization import serialize_model_state


def publish_development_assignment(
    exchange: TrainingExchange,
    client_id: str,
    experiment_id: int,
    round_number: int,
    dataset: str = "pneumonia_mnist",
    local_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.001,
) -> TrainingAssignment:
    model = create_model(dataset)

    state = OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
    )

    assignment = TrainingAssignment(
        experiment_id=experiment_id,
        round_number=round_number,
        dataset=dataset,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_state_base64=encode_model_payload(
            serialize_model_state(state)
        ),
    )

    exchange.publish_assignment(
        client_id=client_id,
        assignment=assignment,
    )

    return assignment
