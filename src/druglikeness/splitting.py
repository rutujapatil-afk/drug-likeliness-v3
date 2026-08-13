from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_SEED = 20260813


def split_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic stratified train/validation/test splits."""
    required = {
        "canonical_smiles",
        "label",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if dataframe["canonical_smiles"].duplicated().any():
        raise ValueError(
            "Dataset contains duplicate canonical SMILES."
        )

    train, temporary = train_test_split(
        dataframe,
        test_size=0.20,
        stratify=dataframe["label"],
        random_state=RANDOM_SEED,
    )

    validation, test = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["label"],
        random_state=RANDOM_SEED,
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )