from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetQCReport:
    """Summary of molecular dataset quality."""

    total_records: int
    valid_records: int
    invalid_records: int
    missing_smiles: int
    duplicate_canonical_smiles: int
    status_counts: dict[str, int]


def generate_qc_report(dataframe: pd.DataFrame) -> DatasetQCReport:
    """
    Generate a quality-control summary for a processed molecular dataset.

    Parameters
    ----------
    dataframe:
        Processed molecular DataFrame.

    Returns
    -------
    DatasetQCReport
        Summary statistics describing dataset quality.
    """
    required_columns = {
        "original_smiles",
        "canonical_smiles",
        "valid",
        "standardization_status",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    total_records = len(dataframe)

    valid_records = int(dataframe["valid"].sum())
    invalid_records = total_records - valid_records

    missing_smiles = int(
        dataframe["original_smiles"].isna().sum()
    )

    duplicate_canonical_smiles = int(
        dataframe.loc[
            dataframe["canonical_smiles"].notna(),
            "canonical_smiles",
        ].duplicated().sum()
    )

    status_counts = (
        dataframe["standardization_status"]
        .value_counts()
        .to_dict()
    )

    return DatasetQCReport(
        total_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        missing_smiles=missing_smiles,
        duplicate_canonical_smiles=duplicate_canonical_smiles,
        status_counts=status_counts,
    )