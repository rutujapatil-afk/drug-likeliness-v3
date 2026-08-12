from pathlib import Path

import pandas as pd

from .processing import process_molecular_record


REQUIRED_COLUMNS = {"source_id", "smiles"}


def load_molecular_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a molecular CSV file.

    The CSV must contain:
        source_id
        smiles

    Parameters
    ----------
    path:
        Path to the input CSV.

    Returns
    -------
    pandas.DataFrame
        Loaded molecular dataset.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    path = Path(path)

    dataframe = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    return dataframe


def process_molecular_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardize all molecular records in a DataFrame.

    Parameters
    ----------
    dataframe:
        DataFrame containing source_id and smiles columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing original and standardized molecular fields.
    """
    records = [
        process_molecular_record(
            source_id=str(row.source_id),
            original_smiles=row.smiles,
        )
        for row in dataframe.itertuples(index=False)
    ]

    return pd.DataFrame(
        [
            {
                "source_id": record.source_id,
                "original_smiles": record.original_smiles,
                "canonical_smiles": record.canonical_smiles,
                "valid": record.valid,
                "standardization_status": record.standardization_status,
            }
            for record in records
        ]
    )