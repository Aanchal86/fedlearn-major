
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results/comparisons")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    # The current summary may contain both datasets.
    paths = [
        RESULTS_DIR / "comparison_summary.csv",
        RESULTS_DIR / "fashion_mnist_comparison_summary.csv",
        RESULTS_DIR / "pneumonia_mnist_comparison_summary.csv",
    ]

    frames = [
        pd.read_csv(path)
        for path in paths
        if path.exists()
    ]

    if not frames:
        raise FileNotFoundError("No comparison CSV files found")

    df = pd.concat(frames, ignore_index=True)

    # Avoid duplicate records from multiple summary files.
    df = df.drop_duplicates(
        subset=["experiment_id"],
        keep="last",
    )

    return df


def plot_metric(df, dataset, metric, ylabel, filename):
    subset = df[df["dataset"] == dataset].copy()

    subset["setting"] = subset.apply(
        lambda row: (
            "IID"
            if row["partition_type"] == "iid"
            else f"Dirichlet α={row['dirichlet_alpha']:g}"
        ),
        axis=1,
    )

    settings = ["IID", "Dirichlet α=0.5", "Dirichlet α=0.1"]
    methods = ["fedavg", "adaptive"]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = range(len(settings))
    width = 0.35

    for i, method in enumerate(methods):
        values = []

        for setting in settings:
            match = subset[
                (subset["setting"] == setting)
                & (subset["aggregation_method"] == method)
            ]

            if len(match) != 1:
                raise ValueError(
                    f"Expected one result: {setting}, {method}"
                )

            values.append(float(match.iloc[0][metric]))

        positions = [
            position + (i - 0.5) * width
            for position in x
        ]

        bars = ax.bar(
            positions,
            values,
            width,
            label=method.upper(),
        )

        ax.bar_label(bars, fmt="%.3f", padding=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(settings)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset} — {ylabel}")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    if metric != "final_loss":
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def main():
    df = load_results()

    for dataset in ["fashion_mnist", "pneumonia_mnist"]:
        plot_metric(
            df, dataset,
            "final_accuracy",
            "Validation accuracy",
            f"{dataset}_accuracy.png",
        )

        plot_metric(
            df, dataset,
            "final_macro_f1",
            "Validation Macro F1",
            f"{dataset}_macro_f1.png",
        )

        plot_metric(
            df, dataset,
            "final_loss",
            "Validation loss",
            f"{dataset}_loss.png",
        )

    print("PASS: Six comparison figures generated")
    print(f"Saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
