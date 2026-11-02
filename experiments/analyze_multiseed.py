
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/multiseed")
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    path = RESULTS_DIR / "multiseed_summary.csv"

    df = pd.read_csv(path)

    required = {
        "dataset", "partition_type", "dirichlet_alpha",
        "aggregation_method", "seed", "final_accuracy",
        "final_macro_f1", "final_loss",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["distribution"] = df.apply(
        lambda row: (
            "IID"
            if row["partition_type"] == "iid"
            else f"Dirichlet alpha={row['dirichlet_alpha']:g}"
        ),
        axis=1,
    )

    keys = [
        "dataset", "distribution",
        "aggregation_method",
    ]

    summary = df.groupby(keys).agg(
        n_seeds=("seed", "nunique"),
        accuracy_mean=("final_accuracy", "mean"),
        accuracy_std=("final_accuracy", "std"),
        macro_f1_mean=("final_macro_f1", "mean"),
        macro_f1_std=("final_macro_f1", "std"),
        loss_mean=("final_loss", "mean"),
        loss_std=("final_loss", "std"),
    ).reset_index()

    # Convert accuracy to percentages.
    summary["accuracy_mean"] *= 100
    summary["accuracy_std"] *= 100

    output_path = RESULTS_DIR / "statistical_summary.csv"
    summary.to_csv(output_path, index=False)

    # Paired comparison: same dataset, distribution, seed.
    paired = df.pivot(
        index=["dataset", "distribution", "seed"],
        columns="aggregation_method",
        values=["final_accuracy", "final_macro_f1"],
    )

    paired = paired.dropna(
        subset=[
            ("final_accuracy", "fedavg"),
            ("final_accuracy", "adaptive"),
            ("final_macro_f1", "fedavg"),
            ("final_macro_f1", "adaptive"),
        ]
    )

    differences = pd.DataFrame({
        "accuracy_difference_pp": (
            paired[("final_accuracy", "adaptive")]
            - paired[("final_accuracy", "fedavg")]
        ) * 100,
        "macro_f1_difference": (
            paired[("final_macro_f1", "adaptive")]
            - paired[("final_macro_f1", "fedavg")]
        ),
    }).reset_index()

    differences.to_csv(
        RESULTS_DIR / "paired_differences.csv",
        index=False,
    )

    paired_summary = differences.groupby(
        ["dataset", "distribution"]
    ).agg(
        n_seeds=("seed", "nunique"),
        mean_accuracy_difference_pp=(
            "accuracy_difference_pp", "mean"
        ),
        std_accuracy_difference_pp=(
            "accuracy_difference_pp", "std"
        ),
        mean_macro_f1_difference=(
            "macro_f1_difference", "mean"
        ),
    ).reset_index()

    paired_summary.to_csv(
        RESULTS_DIR / "paired_summary.csv",
        index=False,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in df["dataset"].unique():
        subset = summary[summary["dataset"] == dataset]

        distributions = [
            "IID",
            "Dirichlet alpha=0.5",
            "Dirichlet alpha=0.1",
        ]

        fig, ax = plt.subplots(figsize=(9, 5))

        for method in ["fedavg", "adaptive"]:
            data = subset[
                subset["aggregation_method"] == method
            ].set_index("distribution")

            data = data.reindex(distributions)

            ax.errorbar(
                distributions,
                data["accuracy_mean"],
                yerr=data["accuracy_std"],
                marker="o",
                capsize=5,
                label=method.upper(),
            )

        ax.set_xlabel("Data distribution")
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title(
            f"{dataset} — Mean accuracy across seeds"
        )
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            FIGURES_DIR / f"{dataset}_multiseed.png",
            dpi=300,
        )
        plt.close(fig)

    print("\n===== STATISTICAL SUMMARY =====")
    print(summary.to_string(index=False))

    print("\n===== PAIRED COMPARISON =====")
    print(paired_summary.to_string(index=False))

    print("\nPASS: Statistical evaluation completed")


if __name__ == "__main__":
    main()
