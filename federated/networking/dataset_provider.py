
from torch.utils.data import Dataset

from federated.data.dataloaders import StandardizedDataset
from federated.data.splits import get_dataset_splits
from federated.partitioning.iid import partition_iid
from federated.partitioning.dirichlet import partition_dirichlet
from federated.partitioning.label_noise import NoisyLabelDataset


def get_client_dataset(
    dataset_name: str,
    client_id: str,
    num_clients: int = 5,
    partition_type: str = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    noise_rate: float = 0.0,
    noisy_client_id: int | None = None,
) -> Dataset:
    if dataset_name not in ("fashion_mnist", "pneumonia_mnist"):
        raise ValueError("Unsupported dataset")

    if num_clients < 1:
        raise ValueError("num_clients must be positive")

    if not client_id.startswith("client_"):
        raise ValueError("Invalid client ID format")

    try:
        client_number = int(client_id.removeprefix("client_"))
    except ValueError as exc:
        raise ValueError("Invalid client ID format") from exc

    if not 1 <= client_number <= num_clients:
        raise ValueError("Client ID is outside configured range")

    if not 0 <= noise_rate <= 1:
        raise ValueError("noise_rate must be between 0 and 1")

    if noise_rate > 0 and (
        noisy_client_id is None
        or not 1 <= noisy_client_id <= num_clients
    ):
        raise ValueError("Invalid noisy client ID")

    splits = get_dataset_splits(
        dataset_name,
        seed=seed,
    )

    train_dataset = StandardizedDataset(splits.train)

    if partition_type == "iid":
        partitions = partition_iid(
            train_dataset,
            num_clients=num_clients,
            seed=seed,
        )

    elif partition_type == "dirichlet":
        partitions = partition_dirichlet(
            train_dataset,
            num_clients=num_clients,
            alpha=dirichlet_alpha,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unsupported partition type: {partition_type}"
        )

    client_index = client_number - 1
    client_dataset = partitions[client_index]

    if noise_rate > 0 and client_number == noisy_client_id:
        num_classes = (
            10 if dataset_name == "fashion_mnist" else 2
        )

        client_dataset = NoisyLabelDataset(
            dataset=client_dataset,
            noise_rate=noise_rate,
            num_classes=num_classes,
            seed=seed + 10000 + client_index,
        )

    return client_dataset
