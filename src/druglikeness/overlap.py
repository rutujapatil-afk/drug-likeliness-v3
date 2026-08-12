import pandas as pd


REQUIRED_COLUMNS = {
    "canonical_smiles",
}


def _validate_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Validate that a DataFrame contains required columns."""
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")


def find_duplicate_molecules(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find duplicate molecular identities within a dataset.

    Duplicate identity is determined using canonical_smiles.

    Returns
    -------
    pandas.DataFrame
        Rows whose canonical SMILES occur more than once.
    """
    _validate_columns(dataframe, REQUIRED_COLUMNS)

    valid_molecules = dataframe[
        dataframe["canonical_smiles"].notna()
    ]

    duplicate_mask = valid_molecules[
        "canonical_smiles"
    ].duplicated(keep=False)

    return valid_molecules.loc[duplicate_mask].copy()


def find_cross_dataset_overlap(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find molecular identities shared between two datasets.

    Parameters
    ----------
    first:
        First molecular dataset.

    second:
        Second molecular dataset.

    Returns
    -------
    pandas.DataFrame
        Canonical SMILES present in both datasets.
    """
    _validate_columns(first, REQUIRED_COLUMNS)
    _validate_columns(second, REQUIRED_COLUMNS)

    first_smiles = set(
        first["canonical_smiles"].dropna()
    )

    second_smiles = set(
        second["canonical_smiles"].dropna()
    )

    overlap = sorted(first_smiles & second_smiles)

    return pd.DataFrame(
        {"canonical_smiles": overlap}
    )


def find_conflicting_labels(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find molecular identities associated with multiple labels.

    Parameters
    ----------
    dataframe:
        DataFrame containing canonical_smiles and label columns.

    Returns
    -------
    pandas.DataFrame
        Molecular identities associated with more than one label.
    """
    _validate_columns(
        dataframe,
        {"canonical_smiles", "label"},
    )

    valid_molecules = dataframe[
        dataframe["canonical_smiles"].notna()
    ]

    label_counts = (
        valid_molecules
        .groupby("canonical_smiles")["label"]
        .nunique()
    )

    conflicting_smiles = label_counts[
        label_counts > 1
    ].index

    return valid_molecules[
        valid_molecules["canonical_smiles"].isin(
            conflicting_smiles
        )
    ].copy()