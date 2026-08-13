from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

from .molecules import smiles_to_molecule
from .processing import process_molecular_record
from .records import MolecularRecord

EXPECTED_COLUMNS = [
    "SMILES",
    "InChI",
    "InChIKey",
    "ID",
    "INN",
    "CAS_RN",
]


def load_drugcentral(
    path: str | Path,
) -> pd.DataFrame:
    """Load a DrugCentral structures TSV file.

    Parameters
    ----------
    path:
        Path to the DrugCentral structures.smiles.tsv file.

    Returns
    -------
    pandas.DataFrame
        DrugCentral records containing the expected columns.
    """
    path = Path(path)

    dataframe = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "SMILES": "string",
            "InChI": "string",
            "InChIKey": "string",
            "ID": "Int64",
            "INN": "string",
            "CAS_RN": "string",
        },
    )

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "DrugCentral file is missing expected columns: "
            f"{missing_columns}"
        )

    return dataframe[EXPECTED_COLUMNS].copy()


def process_drugcentral(
    dataframe: pd.DataFrame,
) -> list[MolecularRecord]:
    """Convert DrugCentral records to project molecular records.

    DrugCentral-specific metadata is used to identify each source
    record, while molecular validation and standardization are
    delegated to the project's common processing pipeline.
    """
    records: list[MolecularRecord] = []

    for row in dataframe.itertuples(index=False):
        record = process_molecular_record(
            source_id=str(row.ID),
            original_smiles=str(row.SMILES),
        )

        records.append(record)

    return records


def summarize_drugcentral(
    records: list[MolecularRecord],
) -> dict[str, int | str]:
    """Summarize processed DrugCentral molecular records."""
    total_records = len(records)

    valid_records = sum(
        record.valid
        for record in records
    )

    invalid_records = total_records - valid_records

    canonical_smiles = [
        record.canonical_smiles
        for record in records
        if record.canonical_smiles is not None
    ]

    unique_canonical_smiles = len(
        set(canonical_smiles)
    )

    duplicate_canonical_smiles = (
        len(canonical_smiles)
        - unique_canonical_smiles
    )

    return {
        "source": "DrugCentral",
        "records": total_records,
        "valid_smiles": valid_records,
        "invalid_smiles": invalid_records,
        "canonical_smiles": len(canonical_smiles),
        "unique_canonical_smiles": unique_canonical_smiles,
        "duplicate_canonical_smiles": duplicate_canonical_smiles,
    }