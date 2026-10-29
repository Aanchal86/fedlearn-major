
import copy

import torch
from torch.utils.data import Subset

from federated.client.trainer import ClientTrainer
from federated.data.dataloaders import StandardizedDataset
from federated.data.splits import get_dataset_splits
from federated.models.cnn import create_model
from federated.partitioning.iid import partition_iid


def verify_dataset(dataset_name):
    print(f"\n{'=' * 55}")
    print(f"Dataset: {dataset_name}")
    print("=" * 55)

    torch.manual_seed(42)

    splits = get_dataset_splits(dataset_name, seed=42)
    train_dataset = StandardizedDataset(splits.train)

    clients = partition_iid(
        train_dataset,
        num_clients=5,
        seed=42,
    )

    global_model = create_model(dataset_name)
    original_state = copy.deepcopy(global_model.state_dict())

    for client_id, client_dataset in clients.items():
        # Use 64 samples per client for quick verification.
        smoke_dataset = Subset(
            client_dataset,
            range(min(64, len(client_dataset))),
        )

        trainer = ClientTrainer(
            client_id=f"client_{client_id + 1}",
            dataset=smoke_dataset,
            batch_size=16,
            seed=42 + client_id,
        )

        update = trainer.train(
            global_model=global_model,
            round_number=1,
            local_epochs=1,
            learning_rate=0.001,
            device="cpu",
        )

        changed = any(
            not torch.equal(update.model_state[name], original_state[name])
            for name in original_state
        )

        assert changed, "Local training did not update weights"
        assert update.num_samples == len(smoke_dataset)
        update.validate()

        print(
            f"{update.client_id}: "
            f"samples={update.num_samples}, "
            f"loss={update.training_loss:.4f}, "
            f"accuracy={update.training_accuracy:.4f}, "
            f"time={update.training_time_seconds:.2f}s"
        )

    for name, tensor in global_model.state_dict().items():
        assert torch.equal(tensor, original_state[name])

    print("PASS: All five clients trained independently.")
    print("PASS: Global model weights remained unchanged.")


if __name__ == "__main__":
    for dataset in ("fashion_mnist", "pneumonia_mnist"):
        verify_dataset(dataset)
