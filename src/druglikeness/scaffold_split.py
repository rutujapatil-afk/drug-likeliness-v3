from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


RANDOM_SEED = 20260813


def scaffold_from_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
    )


def scaffold_split(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split molecules by Bemis-Murcko scaffold."""
    required = {"canonical_smiles", "label"}

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

    dataframe["scaffold"] = dataframe[
        "canonical_smiles"
    ].map(scaffold_from_smiles)

    groups = (
        dataframe.groupby("scaffold", sort=True)
        .indices
    )

    scaffold_groups = list(groups.items())

    # Deterministic ordering followed by deterministic shuffling.
    import random

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(scaffold_groups)

    total = len(dataframe)

    train_target = int(total * 0.80)
    validation_target = int(total * 0.10)

    train_indices = []
    validation_indices = []
    test_indices = []

    for _, indices in scaffold_groups:
        if len(train_indices) < train_target:
            train_indices.extend(indices)
        elif len(validation_indices) < validation_target:
            validation_indices.extend(indices)
        else:
            test_indices.extend(indices)

    train = dataframe.loc[train_indices].copy()
    validation = dataframe.loc[validation_indices].copy()
    test = dataframe.loc[test_indices].copy()

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )