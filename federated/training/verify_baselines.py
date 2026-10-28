
import csv
import json
from pathlib import Path

import torch

from federated.data.dataloaders import create_dataloaders
from federated.models.cnn import create_model
from federated.training.evaluator import evaluate_model


DATASETS = ["fashion_mnist", "pneumonia_mnist"]

RESULT_DIR = Path("results/baselines")
CHECKPOINT_DIR = Path("checkpoints")


def verify_baseline(dataset_name: str) -> dict:
    result_path = RESULT_DIR / f"{dataset_name}_centralized.json"
    checkpoint_path = (
        CHECKPOINT_DIR / f"{dataset_name}_centralized_best.pt"
    )

    if not result_path.exists():
        raise FileNotFoundError(result_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    with result_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["dataset"] == dataset_name

    model = create_model(dataset_name)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, val_loader, _ = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=results["batch_size"],
        seed=results["seed"],
    )

    metrics = evaluate_model(
        model=model,
        data_loader=val_loader,
        device="cpu",
    )

    recorded_f1 = results["best_validation_macro_f1"]
    verified_f1 = metrics["macro_f1"]

    if abs(recorded_f1 - verified_f1) > 1e-6:
        raise AssertionError(
            f"{dataset_name}: checkpoint mismatch. "
            f"Recorded F1={recorded_f1:.6f}, "
            f"Verified F1={verified_f1:.6f}"
        )

    print(f"\nDataset: {dataset_name}")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Verified accuracy: {metrics['accuracy']:.4f}")
    print(f"Verified macro-F1: {verified_f1:.4f}")
    print("Checkpoint verification: PASSED")

    return {
        "dataset": dataset_name,
        "model": "small_cnn",
        "training_type": "centralized",
        "best_epoch": results["best_epoch"],
        "validation_loss": metrics["loss"],
        "validation_accuracy": metrics["accuracy"],
        "validation_macro_precision": metrics["macro_precision"],
        "validation_macro_recall": metrics["macro_recall"],
        "validation_macro_f1": metrics["macro_f1"],
        "validation_weighted_f1": metrics["weighted_f1"],
    }


def main():
    rows = []

    for dataset_name in DATASETS:
        rows.append(verify_baseline(dataset_name))

    output_path = RESULT_DIR / "centralized_comparison.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print("CENTRALIZED BASELINE COMPARISON")
    print("=" * 60)

    for row in rows:
        print(
            f"{row['dataset']:20s} | "
            f"Accuracy: {row['validation_accuracy']:.4f} | "
            f"Macro-F1: {row['validation_macro_f1']:.4f}"
        )

    print(f"\nComparison saved: {output_path}")


if __name__ == "__main__":
    main()
