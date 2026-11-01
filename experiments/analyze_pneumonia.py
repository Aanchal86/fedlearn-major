
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from federated.contracts.experiment import ExperimentConfig
from federated.data.dataloaders import StandardizedDataset
from federated.data.splits import get_dataset_splits
from federated.server.coordinator import FederatedCoordinator


OUTPUT_DIR = Path("results/error_analysis")


def analyze_model(model, dataset, device="cpu"):
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for index in range(len(dataset)):
            image, label = dataset[index]

            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)

            image = image.unsqueeze(0).to(device)

            logits = model(image)
            predicted = torch.argmax(logits, dim=1).item()

            predictions.append(predicted)
            true_labels.append(int(np.asarray(label).item()))

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1],
    )

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
        output_dict=True,
    )

    return matrix, report, true_labels, predictions


def run_analysis(method):
    config = ExperimentConfig(
        dataset="pneumonia_mnist",
        num_clients=5,
        num_rounds=10,
        local_epochs=1,
        batch_size=64,
        learning_rate=0.001,
        partition_type="dirichlet",
        dirichlet_alpha=0.1,
        aggregation_method=method,
        random_seed=42,
        device="cpu",
    )

    print(f"\nTraining: {method}")

    coordinator = FederatedCoordinator(config)
    coordinator.run()

    # Use validation data, matching the Day 13 metrics.
    splits = get_dataset_splits(
        "pneumonia_mnist",
        seed=42,
    )

    val_dataset = StandardizedDataset(splits.val)

    matrix, report, labels, predictions = analyze_model(
        coordinator.global_model,
        val_dataset,
    )

    output = OUTPUT_DIR / method
    output.mkdir(parents=True, exist_ok=True)

    with open(
        output / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    with open(
        output / "predictions.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", "predicted_label"])
        writer.writerows(zip(labels, predictions))

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(matrix, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[i, j] > matrix.max() / 2
                    else "black"
                ),
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        f"PneumoniaMNIST α=0.1 — {method.upper()}"
    )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    fig.savefig(
        output / "confusion_matrix.png",
        dpi=300,
    )
    plt.close(fig)

    print("\nConfusion matrix:")
    print(matrix)

    print("\nPer-class F1:")
    for label in ["0", "1"]:
        print(
            f"Class {label}: "
            f"{report[label]['f1-score']:.4f}"
        )

    print(
        "Predicted class distribution:",
        np.bincount(predictions, minlength=2).tolist(),
    )


if __name__ == "__main__":
    for aggregation in ["fedavg", "adaptive"]:
        run_analysis(aggregation)

    print("\nPASS: PneumoniaMNIST error analysis completed")
