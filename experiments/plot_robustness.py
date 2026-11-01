
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results/robustness")
FIGURES_DIR = RESULTS_DIR / "figures"


def load_round_metrics():
    files = sorted(
        RESULTS_DIR.glob("*/round_metrics.csv")
    )

    if not files:
        raise FileNotFoundError(
            "No robustness round_metrics.csv files found"
        )

    frames = []

    for file in files:
        df = pd.read_csv(file)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def plot_round_metric(df, metric, ylabel, filename):
    fig, ax = plt.subplots(figsize=(9, 5))

    for (noise, method), group in df.groupby(
        ["noise_rate", "aggregation_method"]
    ):
        group = group.sort_values("round")

        label = f"{method.upper()} | {noise:.0%} noise"

        ax.plot(
            group["round"],
            group[metric],
            marker="o",
            label=label,
        )

    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(f"FedLearn Robustness — {ylabel}")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_xticks(sorted(df["round"].unique()))

    if metric in ["accuracy", "macro_f1", "client_3_weight"]:
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_round_metrics()

    plot_round_metric(
        df,
        "accuracy",
        "Global validation accuracy",
        "robustness_accuracy.png",
    )

    plot_round_metric(
        df,
        "macro_f1",
        "Global validation Macro F1",
        "robustness_macro_f1.png",
    )

    plot_round_metric(
        df,
        "client_3_weight",
        "Client 3 aggregation weight",
        "noisy_client_weights.png",
    )

    print("PASS: Robustness figures generated")
    print(f"Saved in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
