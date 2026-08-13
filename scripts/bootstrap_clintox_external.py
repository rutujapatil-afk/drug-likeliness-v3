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
    / "clintox_external_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "clintox_external_bootstrap.csv"
)

SEED = 20260813
N_BOOTSTRAPS = 5000


def main():
    dataframe = pd.read_csv(INPUT_PATH)

    y = dataframe["FDA_APPROVED"].to_numpy()
    probabilities = dataframe[
        "predicted_probability"
    ].to_numpy()

    rng = np.random.default_rng(SEED)

    values = {
        "roc_auc": [],
        "pr_auc": [],
        "accuracy": [],
        "f1": [],
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

        if len(np.unique(y_boot)) < 2:
            continue

        predictions = (
            p_boot >= 0.5
        ).astype(int)

        values["roc_auc"].append(
            roc_auc_score(
                y_boot,
                p_boot,
            )
        )

        values["pr_auc"].append(
            average_precision_score(
                y_boot,
                p_boot,
            )
        )

        values["accuracy"].append(
            accuracy_score(
                y_boot,
                predictions,
            )
        )

        values["f1"].append(
            f1_score(
                y_boot,
                predictions,
                zero_division=0,
            )
        )

        successful += 1

    rows = []

    for metric, bootstrap_values in values.items():
        bootstrap_values = np.asarray(
            bootstrap_values
        )

        if metric == "roc_auc":
            point = roc_auc_score(
                y,
                probabilities,
            )
        elif metric == "pr_auc":
            point = average_precision_score(
                y,
                probabilities,
            )
        elif metric == "accuracy":
            point = accuracy_score(
                y,
                probabilities >= 0.5,
            )
        else:
            point = f1_score(
                y,
                probabilities >= 0.5,
                zero_division=0,
            )

        rows.append(
            {
                "metric": metric,
                "point_estimate": point,
                "bootstrap_mean": bootstrap_values.mean(),
                "ci_lower": np.percentile(
                    bootstrap_values,
                    2.5,
                ),
                "ci_upper": np.percentile(
                    bootstrap_values,
                    97.5,
                ),
                "bootstrap_sd": bootstrap_values.std(
                    ddof=1
                ),
            }
        )

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== ClinTox external bootstrap ===")
    print("Test molecules:", len(dataframe))
    print(
        "Approved:",
        int(y.sum()),
    )
    print(
        "Non-approved:",
        int(len(y) - y.sum()),
    )
    print(
        "Bootstrap replicates:",
        N_BOOTSTRAPS,
    )
    print()

    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()