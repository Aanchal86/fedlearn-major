
import argparse
import csv
import json
from pathlib import Path

from federated.contracts.experiment import ExperimentConfig
from federated.server.coordinator import FederatedCoordinator


OUTPUT_DIR = Path("results/robustness")


def run_robustness_experiment(
    aggregation_method,
    noise_rate,
    rounds=10,
    seed=42,
    device="cpu",
    output_dir=OUTPUT_DIR,
):
    config = ExperimentConfig(
        dataset="fashion_mnist",
        num_clients=5,
        num_rounds=rounds,
        local_epochs=1,
        batch_size=64,
        learning_rate=0.001,
        partition_type="dirichlet",
        dirichlet_alpha=0.5,
        aggregation_method=aggregation_method,
        noise_rate=noise_rate,
        noisy_client_id=3,
        random_seed=seed,
        device=device,
    )

    experiment_id = (
        f"fashion_mnist_dirichlet_0.5_"
        f"{aggregation_method}_noise_{noise_rate:g}_"
        f"rounds_{rounds}_seed_{seed}"
    )

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {experiment_id}")
    print("=" * 65)

    coordinator = FederatedCoordinator(config)
    history = coordinator.run()

    output_dir = Path(output_dir)
    experiment_dir = output_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    round_records = []

    for result in history:
        round_records.append({
            "experiment_id": experiment_id,
            "aggregation_method": aggregation_method,
            "noise_rate": noise_rate,
            "round": result.round_number,
            "accuracy": result.global_accuracy,
            "macro_f1": result.global_macro_f1,
            "loss": result.global_loss,
            "client_3_weight": result.client_weights["client_3"],
            "round_time_seconds": result.round_time_seconds,
        })

    with open(
        experiment_dir / "results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "configuration": config.model_dump(),
                "noise_metadata": coordinator.noise_metadata,
                "history": [
                    result.model_dump()
                    for result in history
                ],
                "client_history": coordinator.client_history,
            },
            file,
            indent=2,
        )

    with open(
        experiment_dir / "round_metrics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=round_records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(round_records)

    final = history[-1]

    return {
        "experiment_id": experiment_id,
        "aggregation_method": aggregation_method,
        "noise_rate": noise_rate,
        "final_accuracy": final.global_accuracy,
        "final_macro_f1": final.global_macro_f1,
        "final_loss": final.global_loss,
        "final_client_3_weight": final.client_weights["client_3"],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []

    for noise_rate in [0.3, 0.6]:
        for method in ["fedavg", "adaptive"]:
            summary = run_robustness_experiment(
                aggregation_method=method,
                noise_rate=noise_rate,
                rounds=args.rounds,
                seed=args.seed,
                device=args.device,
            )

            summaries.append(summary)

            # Preserve successful results after each run.
            with open(
                OUTPUT_DIR / "robustness_summary.csv",
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=summaries[0].keys(),
                )
                writer.writeheader()
                writer.writerows(summaries)

    print("\nPASS: Robustness experiments completed")


if __name__ == "__main__":
    main()
