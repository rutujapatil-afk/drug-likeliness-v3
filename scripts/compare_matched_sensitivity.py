from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "repeated_scaffold_results.csv"
)

MATCHED_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "caliper_matched_training_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "matched_sensitivity_comparison.csv"
)


def mean_ci(values):
    values = np.asarray(values, dtype=float)

    mean = values.mean()
    sd = values.std(ddof=1)

    se = sd / np.sqrt(len(values))

    # t critical value for n=5, df=4
    t_critical = 2.776445

    lower = mean - t_critical * se
    upper = mean + t_critical * se

    return mean, sd, lower, upper


def main():
    original = pd.read_csv(
        ORIGINAL_PATH
    )

    matched = pd.read_csv(
        MATCHED_PATH
    )

    matched = matched[
        matched["split"] == "test"
    ].copy()

    # Keep only the columns required for comparison.
    matched = matched[
        [
            "seed",
            "roc_auc",
            "pr_auc",
        ]
    ].rename(
        columns={
            "roc_auc": "matched_roc_auc",
            "pr_auc": "matched_pr_auc",
        }
    )

    comparison = original.merge(
        matched,
        on="seed",
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != len(matched):
        raise ValueError(
            "Seed matching failed: expected one matched result per seed."
        )

    comparison["delta_roc_auc"] = (
        comparison["matched_roc_auc"]
        - comparison["rf_roc_auc"]
    )

    comparison["delta_pr_auc"] = (
        comparison["matched_pr_auc"]
        - comparison["rf_pr_auc"]
    )

    comparison["relative_roc_change_pct"] = (
        comparison["delta_roc_auc"]
        / comparison["rf_roc_auc"]
        * 100
    )

    comparison["relative_pr_change_pct"] = (
        comparison["delta_pr_auc"]
        / comparison["rf_pr_auc"]
        * 100
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "=== Matched sensitivity comparison ==="
    )

    print(
        comparison[
            [
                "seed",
                "rf_roc_auc",
                "matched_roc_auc",
                "delta_roc_auc",
                "rf_pr_auc",
                "matched_pr_auc",
                "delta_pr_auc",
            ]
        ].round(4).to_string(
            index=False
        )
    )

    print("\n=== Paired differences ===")

    for metric, column in [
        (
            "ROC-AUC",
            "delta_roc_auc",
        ),
        (
            "PR-AUC",
            "delta_pr_auc",
        ),
    ]:
        mean, sd, lower, upper = mean_ci(
            comparison[column]
        )

        print(
            f"{metric}: "
            f"mean Δ={mean:.4f}, "
            f"SD={sd:.4f}, "
            f"95% CI={lower:.4f} to {upper:.4f}"
        )

    print("\nRelative changes:")

    print(
        "ROC-AUC:",
        f"{comparison['relative_roc_change_pct'].mean():.2f}%"
    )

    print(
        "PR-AUC:",
        f"{comparison['relative_pr_change_pct'].mean():.2f}%"
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()