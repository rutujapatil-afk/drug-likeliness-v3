from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "error_group_statistics.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "error_group_statistics_corrected.csv"
)


def benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction."""
    p_values = list(p_values)
    n = len(p_values)

    order = sorted(
        range(n),
        key=lambda i: p_values[i],
    )

    adjusted = [0.0] * n
    running_min = 1.0

    for rank in range(n, 0, -1):
        index = order[rank - 1]
        adjusted_value = (
            p_values[index] * n / rank
        )

        running_min = min(
            running_min,
            adjusted_value,
        )

        adjusted[index] = min(
            running_min,
            1.0,
        )

    return adjusted


def main():
    dataframe = pd.read_csv(INPUT_PATH)

    dataframe["p_fdr_bh"] = float("nan")

    for comparison in dataframe[
        ["group_a", "group_b"]
    ].drop_duplicates().itertuples(index=False):

        mask = (
            (dataframe["group_a"] == comparison.group_a)
            & (dataframe["group_b"] == comparison.group_b)
        )

        p_values = dataframe.loc[
            mask,
            "mannwhitney_p",
        ].tolist()

        corrected = benjamini_hochberg(
            p_values
        )

        dataframe.loc[
            mask,
            "p_fdr_bh",
        ] = corrected

    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== FDR-corrected error statistics ===")

    print(
        dataframe[
            [
                "descriptor",
                "group_a",
                "group_b",
                "mannwhitney_p",
                "p_fdr_bh",
                "cliffs_delta",
            ]
        ].to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()