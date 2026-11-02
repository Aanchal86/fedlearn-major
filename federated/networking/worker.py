
from torch.utils.data import Dataset

from federated.client.trainer import ClientTrainer
from federated.contracts.client_update import ClientUpdate
from federated.models.cnn import create_model
from federated.networking.protocol import (
    ClientUpdateMessage,
    TrainingAssignment,
    decode_model_payload,
    encode_model_payload,
)
from federated.networking.serialization import (
    deserialize_model_state,
    serialize_model_state,
)


class RemoteClientWorker:
    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        seed: int = 42,
    ):
        self.trainer = ClientTrainer(
            client_id=client_id,
            dataset=dataset,
            seed=seed,
        )

    def execute(
        self,
        assignment: TrainingAssignment,
    ) -> ClientUpdateMessage:
        model = create_model(assignment.dataset)

        global_state = deserialize_model_state(
            decode_model_payload(
                assignment.model_state_base64
            )
        )

        model.load_state_dict(global_state, strict=True)

        trainer = ClientTrainer(
            client_id=self.trainer.client_id,
            dataset=self.trainer.dataset,
            batch_size=assignment.batch_size,
            seed=self.trainer.seed,
        )

        update: ClientUpdate = trainer.train(
            global_model=model,
            round_number=assignment.round_number,
            local_epochs=assignment.local_epochs,
            learning_rate=assignment.learning_rate,
            device="cpu",
        )

        return ClientUpdateMessage(
            experiment_id=assignment.experiment_id,
            client_id=update.client_id,
            round_number=update.round_number,
            num_samples=update.num_samples,
            training_loss=update.training_loss,
            training_accuracy=update.training_accuracy,
            training_time_seconds=update.training_time_seconds,
            model_state_base64=encode_model_payload(
                serialize_model_state(update.model_state)
            ),
        )
