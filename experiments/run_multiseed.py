
import argparse
import csv
import json
from pathlib import Path

from federated.contracts.experiment import ExperimentConfig
from experiments.compare_aggregations import (
    SCENARIOS,
    build_experiment_id,
    run_single_experiment,
)


OUTPUT_DIR = Path("results/multiseed")


def run_multiseed(
    datasets,
    seeds,
    rounds=10,
    device="cpu",
    output_dir=OUTPUT_DIR,
    scenarios=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = SCENARIOS if scenarios is None else scenarios

    summaries = []

    for dataset in datasets:
        for partition, alpha, aggregation in scenarios:
            for seed in seeds:
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

                experiment_id = build_experiment_id(config)
                result_file = (
                    output_dir
                    / experiment_id
                    / "results.json"
                )

                if result_file.exists():
                    print(f"SKIP: {experiment_id}")

                    with open(
                        result_file,
                        "r",
                        encoding="utf-8",
                    ) as file:
                        saved = json.load(file)

                    if saved.get("configuration") != config.model_dump():
                        raise ValueError(
                            "Existing results have a different "
                            "experiment configuration"
                        )

                    summary = saved["summary"]
                else:
                    summary = run_single_experiment(
                        config,
                        output_dir=output_dir,
                    )

                summaries.append(summary)

                # Save a cumulative summary after each run.
                summary_path = (
                    output_dir / "multiseed_summary.csv"
                )

                with open(
                    summary_path,
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

    print("\nPASS: Multi-seed experiments completed")

    return summaries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "pneumonia_mnist", "both"],
        default="fashion_mnist",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )

    parser.add_argument("--rounds", type=int, default=10)

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
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

    run_multiseed(
        datasets=datasets,
        seeds=(
            [42] if args.smoke_test else args.seeds
        ),
        rounds=(
            1 if args.smoke_test else args.rounds
        ),
        device=args.device,
        scenarios=scenarios,
    )


if __name__ == "__main__":
    main()
