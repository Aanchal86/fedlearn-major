
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from federated.data.dataloaders import create_dataloaders
from federated.models.cnn import create_model
from federated.training.trainer import train_one_epoch
from federated.training.evaluator import evaluate_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_baseline(
    dataset_name: str,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
    device: str = "cpu",
):
    if epochs < 1:
        raise ValueError("epochs must be positive")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    set_seed(seed)

    train_loader, val_loader, _ = create_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        seed=seed,
    )

    model = create_model(dataset_name).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = torch.nn.CrossEntropyLoss()

    history = []
    best_macro_f1 = -1.0
    best_epoch = None

    checkpoint_dir = Path("checkpoints")
    result_dir = Path("results/baselines")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / (
        f"{dataset_name}_centralized_best.pt"
    )

    print(f"\nDataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device(device),
        )

        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(epoch_result)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "validation_macro_f1": best_macro_f1,
                },
                checkpoint_path,
            )

    results = {
        "dataset": dataset_name,
        "training_type": "centralized",
        "model": "small_cnn",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "device": device,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }

    result_path = result_dir / (
        f"{dataset_name}_centralized.json"
    )

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("-" * 60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro-F1: {best_macro_f1:.4f}")
    print(f"Checkpoint saved: {checkpoint_path}")
    print(f"Results saved: {result_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run centralized CNN baseline"
    )

    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "pneumonia_mnist"],
        required=True,
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()

    run_baseline(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )
