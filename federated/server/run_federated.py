
import argparse
import json
from pathlib import Path

import torch

from federated.contracts.experiment import ExperimentConfig
from federated.server.coordinator import FederatedCoordinator


def run_experiment(
    dataset_name: str,
    num_clients: int = 5,
    num_rounds: int = 3,
    local_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
    device: str = "cpu",
):
    config = ExperimentConfig(
        dataset=dataset_name,
        num_clients=num_clients,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=seed,
        device=device,
        partition_type="iid",
        aggregation_method="fedavg",
    )

    coordinator = FederatedCoordinator(config)

    print(f"\nDataset: {dataset_name}")
    print(f"Clients: {num_clients}")
    print(f"Rounds: {num_rounds}")
    print(f"Local epochs: {local_epochs}")
    print("-" * 60)

    history = coordinator.run()

    result_dir = Path("results/federated")
    checkpoint_dir = Path("checkpoints")

    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    result_path = result_dir / f"{dataset_name}_fedavg.json"
    checkpoint_path = checkpoint_dir / f"{dataset_name}_fedavg_final.pt"

    results = {
        "dataset": dataset_name,
        "training_type": "federated",
        "model": "small_cnn",
        "configuration": config.model_dump(),
        "history": [
            result.model_dump()
            for result in history
        ],
        "final_validation_accuracy": history[-1].global_accuracy,
        "final_validation_loss": history[-1].global_loss,
        "final_validation_macro_f1": history[-1].global_macro_f1,
        "checkpoint_path": str(checkpoint_path),
    }

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    torch.save(
        {
            "model_state_dict": coordinator.global_model.state_dict(),
            "configuration": config.model_dump(),
            "final_round": num_rounds,
        },
        checkpoint_path,
    )

    print("\n" + "=" * 60)
    print("FEDERATED EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"Final accuracy: {history[-1].global_accuracy:.4f}")
    print(f"Final loss: {history[-1].global_loss:.4f}")
    print(f"Final Macro-F1: {history[-1].global_macro_f1:.4f}")
    print(f"Results saved: {result_path}")
    print(f"Checkpoint saved: {checkpoint_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-round federated learning"
    )

    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "pneumonia_mnist"],
        required=True,
    )
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )
