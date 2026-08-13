from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold


RANDOM_SEED = 20260813


def scaffold_from_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
    )


def scaffold_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create an approximately 80/10/10 scaffold-separated split."""

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

    dataframe = dataframe.copy()

    dataframe["scaffold"] = (
        dataframe["canonical_smiles"]
        .map(scaffold_from_smiles)
    )

    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    groups = dataframe["scaffold"]

    folds = list(
        splitter.split(
            dataframe,
            dataframe["label"],
            groups,
        )
    )

    # First fold -> validation
    # Second fold -> test
    validation_indices = folds[0][1]
    test_indices = folds[1][1]

    validation_set = set(validation_indices)
    test_set = set(test_indices)

    train_indices = [
        index
        for index in dataframe.index
        if index not in validation_set
        and index not in test_set
    ]

    train = dataframe.loc[train_indices].copy()
    validation = dataframe.loc[validation_indices].copy()
    test = dataframe.loc[test_indices].copy()

    return (
        train.drop(columns=["scaffold"]).reset_index(drop=True),
        validation.drop(columns=["scaffold"]).reset_index(drop=True),
        test.drop(columns=["scaffold"]).reset_index(drop=True),
    )