from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "scaffold_test_predictions.csv"
)

DESCRIPTORS = [
    "molecular_weight",
    "heavy_atoms",
    "hbd",
    "hba",
    "rotatable_bonds",
    "rings",
    "aromatic_rings",
    "fraction_csp3",
    "logp",
]


def cliffs_delta(x, y):
    """Estimate Cliff's delta: P(x > y) - P(x < y)."""
    x = list(x)
    y = list(y)

    greater = 0
    less = 0

    for a in x:
        for b in y:
            if a > b:
                greater += 1
            elif a < b:
                less += 1

    total = len(x) * len(y)

    if total == 0:
        return float("nan")

    return (greater - less) / total


def compare_groups(dataframe, group_a, group_b):
    rows = []

    for descriptor in DESCRIPTORS:
        a = dataframe.loc[
            dataframe["error_type"] == group_a,
            descriptor,
        ].dropna()

        b = dataframe.loc[
            dataframe["error_type"] == group_b,
            descriptor,
        ].dropna()

        statistic, p_value = mannwhitneyu(
            a,
            b,
            alternative="two-sided",
        )

        delta = cliffs_delta(a, b)

        rows.append(
            {
                "descriptor": descriptor,
                "group_a": group_a,
                "group_b": group_b,
                "group_a_median": a.median(),
                "group_b_median": b.median(),
                "mannwhitney_p": p_value,
                "cliffs_delta": delta,
            }
        )

    return pd.DataFrame(rows)


def main():
    dataframe = pd.read_csv(INPUT_PATH)

    print("=== Error-group statistical analysis ===")
    print("Rows:", len(dataframe))

    print("\nGroup sizes:")
    print(dataframe["error_type"].value_counts())

    comparisons = [
        ("false_positive", "correct"),
        ("false_negative", "correct"),
        ("false_positive", "false_negative"),
    ]

    all_results = []

    for group_a, group_b in comparisons:
        result = compare_groups(
            dataframe,
            group_a,
            group_b,
        )

        all_results.append(result)

        print(
            f"\n=== {group_a} vs {group_b} ==="
        )

        print(
            result.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    output_path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "error_group_statistics.csv"
    )

    pd.concat(
        all_results,
        ignore_index=True,
    ).to_csv(
        output_path,
        index=False,
    )

    print("\nOutput:")
    print(output_path)


if __name__ == "__main__":
    main()