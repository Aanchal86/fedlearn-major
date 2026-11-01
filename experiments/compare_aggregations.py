
import argparse
import csv
import json
from pathlib import Path

import torch

from federated.contracts.experiment import ExperimentConfig
from federated.server.coordinator import FederatedCoordinator


OUTPUT_DIR = Path("results/comparisons")

SCENARIOS = [
    ("iid", None, "fedavg"),
    ("iid", None, "adaptive"),
    ("dirichlet", 0.5, "fedavg"),
    ("dirichlet", 0.5, "adaptive"),
    ("dirichlet", 0.1, "fedavg"),
    ("dirichlet", 0.1, "adaptive"),
]


def build_experiment_id(config):
    partition = config.partition_type

    if partition == "dirichlet":
        partition += f"_alpha_{config.dirichlet_alpha:g}"

    return (
        f"{config.dataset}_{partition}_"
        f"{config.aggregation_method}_"
        f"seed_{config.random_seed}"
    )


def run_single_experiment(config, output_dir=OUTPUT_DIR):
    experiment_id = build_experiment_id(config)

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: {experiment_id}")
    print("=" * 65)

    coordinator = FederatedCoordinator(config)
    history = coordinator.run()

    final = history[-1]

    round_records = []

    for result in history:
        record = {
            "experiment_id": experiment_id,
            "dataset": config.dataset,
            "partition_type": config.partition_type,
            "dirichlet_alpha": (
                config.dirichlet_alpha
                if config.partition_type == "dirichlet"
                else None
            ),
            "aggregation_method": config.aggregation_method,
            "seed": config.random_seed,
            "round": result.round_number,
            "accuracy": result.global_accuracy,
            "macro_f1": result.global_macro_f1,
            "loss": result.global_loss,
            "round_time_seconds": result.round_time_seconds,
        }

        round_records.append(record)

    summary = {
        "experiment_id": experiment_id,
        "dataset": config.dataset,
        "partition_type": config.partition_type,
        "dirichlet_alpha": (
            config.dirichlet_alpha
            if config.partition_type == "dirichlet"
            else None
        ),
        "aggregation_method": config.aggregation_method,
        "seed": config.random_seed,
        "num_clients": config.num_clients,
        "num_rounds": config.num_rounds,
        "local_epochs": config.local_epochs,
        "final_accuracy": final.global_accuracy,
        "final_macro_f1": final.global_macro_f1,
        "final_loss": final.global_loss,
        "total_round_time_seconds": sum(
            r.round_time_seconds for r in history
        ),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = output_dir / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(
        experiment_dir / "results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "configuration": config.model_dump(),
                "summary": summary,
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

    print(f"\nCompleted: {experiment_id}")
    print(f"Accuracy: {final.global_accuracy:.4f}")
    print(f"Macro F1: {final.global_macro_f1:.4f}")
    print(f"Loss: {final.global_loss:.4f}")

    return summary


def run_comparison(
    datasets,
    rounds=10,
    seed=42,
    device="cpu",
    scenarios=None,
    output_dir=OUTPUT_DIR,
):
    selected_scenarios = (
        SCENARIOS if scenarios is None else scenarios
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    summary_path = output_dir / "comparison_summary.csv"

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            all_summaries = list(reader)

    for dataset in datasets:
        for partition, alpha, aggregation in selected_scenarios:
            config = ExperimentConfig(
                dataset=dataset,
                num_clients=5,
                num_rounds=rounds,
                local_epochs=1,
                batch_size=64,
                learning_rate=0.001,
                partition_type=partition,
                dirichlet_alpha=(
                    alpha if alpha is not None else 0.5
                ),
                aggregation_method=aggregation,
                random_seed=seed,
                device=device,
            )

            summary = run_single_experiment(
                config,
                output_dir=output_dir,
            )

            all_summaries = [
                old
                for old in all_summaries
                if old["experiment_id"] != summary["experiment_id"]
            ]

            all_summaries.append(summary)

            # Save progress after every successful experiment.
            dataset_name = config.dataset
            summary_path = output_dir / f"{dataset_name}_comparison_summary.csv"
            with open(
                summary_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=all_summaries[0].keys(),
                )
                writer.writeheader()
                writer.writerows(all_summaries)

    print("\nAll selected experiments completed.")

    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "pneumonia_mnist", "both"],
        default="fashion_mnist",
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one short IID FedAvg experiment",
    )

    args = parser.parse_args()

    datasets = (
        ["fashion_mnist", "pneumonia_mnist"]
        if args.dataset == "both"
        else [args.dataset]
    )

    scenarios = (
        [SCENARIOS[0]]
        if args.smoke_test
        else SCENARIOS
    )

    run_comparison(
        datasets=datasets,
        rounds=1 if args.smoke_test else args.rounds,
        seed=args.seed,
        device=args.device,
        scenarios=scenarios,
    )
