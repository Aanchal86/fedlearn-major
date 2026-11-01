
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from federated.data.dataloaders import StandardizedDataset
from federated.data.splits import get_dataset_splits
from federated.partitioning.dirichlet import partition_dirichlet
from federated.partitioning.iid import partition_iid


OUTPUT_DIR = Path("results/partitioning")


def generate_heatmap(dataset_name, partition_type, alpha=None):
    splits = get_dataset_splits(dataset_name, seed=42)
    train_dataset = StandardizedDataset(splits.train)

    if partition_type == "iid":
        partitions = partition_iid(
            train_dataset, num_clients=5, seed=42
        )
        name = "iid"
    else:
        partitions = partition_dirichlet(
            train_dataset,
            num_clients=5,
            alpha=alpha,
            seed=42,
        )
        name = f"dirichlet_{alpha}"

    # Collect labels from the training dataset.
    labels = np.array([
        int(np.asarray(train_dataset[i][1]).item())
        for i in range(len(train_dataset))
    ])

    classes = np.unique(labels)

    counts = np.zeros((5, len(classes)), dtype=int)

    for client_id, subset in partitions.items():
        client_labels = labels[subset.indices]

        for j, label in enumerate(classes):
            counts[client_id, j] = np.sum(
                client_labels == label
            )

    fig, ax = plt.subplots(
        figsize=(max(7, len(classes) * 0.85), 4.5)
    )

    image = ax.imshow(counts, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(5))
    ax.set_yticklabels(
        [f"Client {i + 1}" for i in range(5)]
    )

    ax.set_xlabel("Class label")
    ax.set_ylabel("Federated client")

    ax.set_title(
        f"{dataset_name} | "
        f"{partition_type.upper()} "
        f"{'alpha=' + str(alpha) if alpha else ''}"
    )

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            value = counts[i, j]
            ax.text(
                j, i, str(value),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if value > counts.max() / 2
                    else "black"
                ),
            )

    fig.colorbar(image, ax=ax, label="Sample count")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / (
        f"{dataset_name}_{name}.png"
    )

    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"Saved: {output_file}")
    print(f"Client totals: {counts.sum(axis=1).tolist()}")
    print(f"Total allocated: {counts.sum()}")


if __name__ == "__main__":
    for dataset in ["fashion_mnist", "pneumonia_mnist"]:
        generate_heatmap(dataset, "iid")
        generate_heatmap(dataset, "dirichlet", alpha=0.5)
        generate_heatmap(dataset, "dirichlet", alpha=0.1)
