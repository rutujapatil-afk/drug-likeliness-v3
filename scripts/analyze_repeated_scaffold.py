from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "repeated_scaffold_results.csv"
)


def mean_ci(values):
    values = np.asarray(values, dtype=float)

    mean = values.mean()
    sd = values.std(ddof=1)

    # 95% normal-approximation interval.
    # With only 5 repeated splits, report this as descriptive
    # uncertainty rather than a definitive population CI.
    margin = 1.96 * sd / np.sqrt(len(values))

    return mean, sd, mean - margin, mean + margin


def main():
    dataframe = pd.read_csv(INPUT_PATH)

    print("=== Repeated scaffold analysis ===")
    print("Splits:", len(dataframe))

    metrics = [
        "rf_roc_auc",
        "rf_pr_auc",
        "lr_roc_auc",
        "lr_pr_auc",
    ]

    for metric in metrics:
        mean, sd, lower, upper = mean_ci(
            dataframe[metric]
        )

        print(
            f"{metric}: "
            f"mean={mean:.4f}, "
            f"SD={sd:.4f}, "
            f"95% CI={lower:.4f} to {upper:.4f}"
        )

    print("\n=== Paired RF - LR differences ===")

    for rf_metric, lr_metric in [
        ("rf_roc_auc", "lr_roc_auc"),
        ("rf_pr_auc", "lr_pr_auc"),
    ]:
        difference = (
            dataframe[rf_metric]
            - dataframe[lr_metric]
        )

        mean, sd, lower, upper = mean_ci(difference)

        print(
            f"{rf_metric.replace('rf_', '')}: "
            f"mean difference={mean:.4f}, "
            f"SD={sd:.4f}, "
            f"95% CI={lower:.4f} to {upper:.4f}"
        )

        print(
            "Per-seed differences:",
            ", ".join(
                f"{value:.4f}"
                for value in difference
            ),
        )


if __name__ == "__main__":
    main()