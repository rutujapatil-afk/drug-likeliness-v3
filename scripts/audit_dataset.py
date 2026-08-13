from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from pathlib import Path

import pandas as pd

from druglikeness.scaffold_split import (
    scaffold_from_smiles,
    scaffold_split,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)


def main() -> None:
    dataframe = pd.read_csv(DATASET_PATH)

    print("=== Dataset audit ===")
    print("Rows:", len(dataframe))
    print(
        "Unique canonical SMILES:",
        dataframe["canonical_smiles"].nunique(),
    )
    print(
        "Duplicate canonical SMILES:",
        dataframe["canonical_smiles"].duplicated().sum(),
    )

    print("\n=== Cross-class molecular overlap ===")

    positive = set(
        dataframe.loc[
            dataframe["label"] == 1,
            "canonical_smiles",
        ]
    )

    negative = set(
        dataframe.loc[
            dataframe["label"] == 0,
            "canonical_smiles",
        ]
    )

    print(
        "Positive molecules:",
        len(positive),
    )
    print(
        "Negative molecules:",
        len(negative),
    )
    print(
        "Cross-class overlap:",
        len(positive & negative),
    )

    print("\n=== Random split leakage ===")

    train, validation, test = (
        __import__(
            "druglikeness.splitting",
            fromlist=["split_dataset"],
        ).split_dataset(dataframe)
    )

    partitions = {
        "train": set(train["canonical_smiles"]),
        "validation": set(validation["canonical_smiles"]),
        "test": set(test["canonical_smiles"]),
    }

    print(
        "Train/validation overlap:",
        len(partitions["train"] & partitions["validation"]),
    )
    print(
        "Train/test overlap:",
        len(partitions["train"] & partitions["test"]),
    )
    print(
        "Validation/test overlap:",
        len(partitions["validation"] & partitions["test"]),
    )

    print("\n=== Scaffold split audit ===")

    scaffold_dataframe = dataframe.copy()

    scaffold_dataframe["scaffold"] = (
        scaffold_dataframe["canonical_smiles"]
        .map(scaffold_from_smiles)
    )

    scaffold_train, scaffold_validation, scaffold_test = (
        scaffold_split(dataframe)
    )

    scaffold_partitions = {
        "train": set(
            scaffold_dataframe.loc[
                scaffold_dataframe["canonical_smiles"].isin(
                    scaffold_train["canonical_smiles"]
                ),
                "scaffold",
            ]
        ),
        "validation": set(
            scaffold_dataframe.loc[
                scaffold_dataframe["canonical_smiles"].isin(
                    scaffold_validation["canonical_smiles"]
                ),
                "scaffold",
            ]
        ),
        "test": set(
            scaffold_dataframe.loc[
                scaffold_dataframe["canonical_smiles"].isin(
                    scaffold_test["canonical_smiles"]
                ),
                "scaffold",
            ]
        ),
    }

    print(
        "Train/validation scaffold overlap:",
        len(
            scaffold_partitions["train"]
            & scaffold_partitions["validation"]
        ),
    )
    print(
        "Train/test scaffold overlap:",
        len(
            scaffold_partitions["train"]
            & scaffold_partitions["test"]
        ),
    )
    print(
        "Validation/test scaffold overlap:",
        len(
            scaffold_partitions["validation"]
            & scaffold_partitions["test"]
        ),
    )

    print("\n=== Class counts ===")
    print(
        dataframe["label"]
        .value_counts()
        .sort_index()
    )


if __name__ == "__main__":
    main()