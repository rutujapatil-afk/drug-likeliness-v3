from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "scaffold_test_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "bootstrap_test_metrics.csv"
)

SEED = 20260813
N_BOOTSTRAPS = 5000


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "f1": f1_score(
            y_true,
            predictions,
        ),
    }


def main():
    dataframe = pd.read_csv(INPUT_PATH)

    y = dataframe["label"].to_numpy()
    probabilities = dataframe[
        "predicted_probability"
    ].to_numpy()

    rng = np.random.default_rng(SEED)

    metric_names = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "f1",
    ]

    bootstrap_values = {
        metric: []
        for metric in metric_names
    }

    n = len(y)

    successful = 0

    while successful < N_BOOTSTRAPS:
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        y_boot = y[indices]
        p_boot = probabilities[indices]

        # ROC-AUC and PR-AUC require both classes.
        if len(np.unique(y_boot)) < 2:
            continue

        metrics = calculate_metrics(
            y_boot,
            p_boot,
        )

        for metric in metric_names:
            bootstrap_values[metric].append(
                metrics[metric]
            )

        successful += 1

    rows = []

    for metric in metric_names:
        values = np.asarray(
            bootstrap_values[metric]
        )

        rows.append(
            {
                "metric": metric,
                "point_estimate": calculate_metrics(
                    y,
                    probabilities,
                )[metric],
                "bootstrap_mean": values.mean(),
                "ci_lower": np.percentile(
                    values,
                    2.5,
                ),
                "ci_upper": np.percentile(
                    values,
                    97.5,
                ),
                "bootstrap_sd": values.std(
                    ddof=1
                ),
            }
        )

    results = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Bootstrap confidence intervals ===")
    print("Test molecules:", len(dataframe))
    print("Bootstrap replicates:", N_BOOTSTRAPS)
    print()

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()