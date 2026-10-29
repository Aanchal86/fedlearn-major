
import copy

import torch
from torch.utils.data import Subset

from federated.aggregation.fedavg import federated_averaging
from federated.client.trainer import ClientTrainer
from federated.data.dataloaders import StandardizedDataset
from federated.data.splits import get_dataset_splits
from federated.models.cnn import create_model
from federated.partitioning.iid import partition_iid


def verify_fedavg(dataset_name):
    print(f"\n{'=' * 55}")
    print(f"Dataset: {dataset_name}")
    print("=" * 55)

    torch.manual_seed(42)

    splits = get_dataset_splits(dataset_name, seed=42)
    train_dataset = StandardizedDataset(splits.train)

    partitions = partition_iid(
        train_dataset,
        num_clients=5,
        seed=42,
    )

    global_model = create_model(dataset_name)
    initial_state = copy.deepcopy(global_model.state_dict())

    updates = []

    for client_id, dataset in partitions.items():
        smoke_dataset = Subset(
            dataset,
            range(min(64, len(dataset))),
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

        updates.append(update)

        print(
            f"{update.client_id}: "
            f"samples={update.num_samples}, "
            f"loss={update.training_loss:.4f}, "
            f"accuracy={update.training_accuracy:.4f}"
        )

    # Check that local training did not alter the global model.
    for name, tensor in global_model.state_dict().items():
        assert torch.equal(tensor, initial_state[name])

    aggregated_state = federated_averaging(updates)

    # Independently verify the weighted average.
    total_samples = sum(update.num_samples for update in updates)

    for name, tensor in aggregated_state.items():
        if tensor.is_floating_point():
            expected = sum(
                (
                    update.model_state[name].double()
                    * (update.num_samples / total_samples)
                    for update in updates
                ),
                torch.zeros_like(tensor, dtype=torch.float64),
            ).to(tensor.dtype)

            assert torch.allclose(tensor, expected, atol=1e-6)

    global_model.load_state_dict(aggregated_state)

    changed = any(
        not torch.equal(global_model.state_dict()[name], initial_state[name])
        for name in initial_state
    )

    assert changed, "Global model did not change after aggregation"

    print("\nPASS: Five client updates aggregated successfully.")
    print("PASS: Sample-weighted averaging verified.")
    print("PASS: Updated global CNN loaded successfully.")


if __name__ == "__main__":
    for dataset_name in ("fashion_mnist", "pneumonia_mnist"):
        verify_fedavg(dataset_name)
