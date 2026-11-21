from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results/comparisons")
FIGURES_DIR = RESULTS_DIR / "figures"
SUMMARY_PATH = RESULTS_DIR / "fashion_mnist_comparison_summary.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    summary = pd.read_csv(SUMMARY_PATH)

    summary = summary[
        (summary["dataset"] == "fashion_mnist")
        & (summary["seed"] == 42)
        & (summary["num_rounds"] == 10)
    ].copy()

    distributions = [
        ("iid", None, "IID", "iid"),
        ("dirichlet", 0.5, "Non-IID alpha=0.5", "non_iid_05"),
        ("dirichlet", 0.1, "Non-IID alpha=0.1", "non_iid_01"),
    ]

    methods = ["fedavg", "adaptive"]

    for partition, alpha, label, filename in distributions:
        selected = summary[
            summary["partition_type"] == partition
        ]

        if alpha is not None:
            selected = selected[
                selected["dirichlet_alpha"].round(6) == alpha
            ]

        for metric, ylabel in [
            ("accuracy", "Validation Accuracy (%)"),
            ("loss", "Validation Loss"),
        ]:
            fig, ax = plt.subplots(figsize=(9, 5))

            for method in methods:
                match = selected[
                    selected["aggregation_method"] == method
                ]

                if len(match) != 1:
                    raise ValueError(
                        f"Expected one experiment for {label}, {method}; "
                        f"found {len(match)}"
                    )

                experiment_id = match.iloc[0]["experiment_id"]

                metrics_path = (
                    RESULTS_DIR / experiment_id / "round_metrics.csv"
                )

                if not metrics_path.exists():
                    raise FileNotFoundError(metrics_path)

                data = pd.read_csv(metrics_path).sort_values("round")

                values = data[metric]

                if metric == "accuracy":
                    values = values * 100

                ax.plot(
                    data["round"],
                    values,
                    marker="o",
                    linewidth=2,
                    label=method.upper(),
                )

            ax.set_title(f"Fashion-MNIST — {label}")
            ax.set_xlabel("Federated Communication Round")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(1, 11))
            ax.grid(alpha=0.25)
            ax.legend()

            if metric == "accuracy":
                ax.set_ylim(0, 100)

            fig.tight_layout()

            output = FIGURES_DIR / f"{filename}_{metric}_convergence.png"

            fig.savefig(output, dpi=300)
            plt.close(fig)

            print(f"Generated: {output}")

    print("\nPASS: All six convergence figures generated")


if __name__ == "__main__":
    main()
