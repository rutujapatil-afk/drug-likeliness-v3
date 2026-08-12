from pathlib import Path

import pandas as pd
from rdkit import Chem

from .molecules import smiles_to_molecule
from .records import MolecularRecord


EXPECTED_COLUMNS = [
    "SMILES",
    "InChI",
    "InChIKey",
    "ID",
    "INN",
    "CAS_RN",
]


def load_drugcentral(path: str | Path) -> pd.DataFrame:
    """Load a DrugCentral structures TSV file."""
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
    """Convert DrugCentral records to the project molecular record model."""
    records: list[MolecularRecord] = []

    for row in dataframe.itertuples(index=False):
        smiles = str(row.SMILES)

        molecule = smiles_to_molecule(smiles)

        if molecule is None:
            records.append(
                MolecularRecord(
                    source_id=str(row.ID),
                    original_smiles=smiles,
                    canonical_smiles=None,
                    valid=False,
                    standardization_status="invalid_smiles",
                )
            )
            continue

        canonical_smiles = molecule_to_canonical_smiles(molecule)

        records.append(
            MolecularRecord(
                source_id=str(row.ID),
                original_smiles=smiles,
                canonical_smiles=canonical_smiles,
                valid=True,
                standardization_status="standardized",
            )
        )

    return records


def molecule_to_canonical_smiles(molecule) -> str:
    """Convert an RDKit molecule to canonical SMILES."""
    return Chem.MolToSmiles(
        molecule,
        canonical=True,
    )