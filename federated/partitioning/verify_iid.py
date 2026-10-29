
from federated.data.splits import get_dataset_splits
from federated.data.dataloaders import StandardizedDataset
from federated.partitioning.iid import (
    partition_iid,
    get_label_distribution,
)


def verify_dataset(dataset_name):
    splits = get_dataset_splits(dataset_name, seed=42)

    train_dataset = StandardizedDataset(splits.train)
    clients = partition_iid(train_dataset, num_clients=5, seed=42)

    print(f"\nDataset: {dataset_name}")
    print(f"Training samples: {len(splits.train)}")
    print(f"Validation samples: {len(splits.val)}")
    print(f"Test samples: {len(splits.test)}")

    all_indices = []

    for client_id, client_dataset in clients.items():
        all_indices.extend(client_dataset.indices)

        distribution = get_label_distribution(client_dataset)

        print(
            f"Client {client_id + 1}: "
            f"{len(client_dataset)} samples, "
            f"labels={distribution}"
        )

    assert len(all_indices) == len(train_dataset)
    assert len(set(all_indices)) == len(train_dataset)
    assert set(all_indices) == set(range(len(train_dataset)))

    assert all(
        client.dataset is train_dataset
        for client in clients.values()
    )

    print("PASS: All training samples assigned exactly once.")
    print("PASS: Only the training dataset was partitioned.")


if __name__ == "__main__":
    for name in ("fashion_mnist", "pneumonia_mnist"):
        verify_dataset(name)
