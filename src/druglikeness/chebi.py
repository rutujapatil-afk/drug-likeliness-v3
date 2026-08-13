from __future__ import annotations

import gzip
from pathlib import Path

from rdkit import Chem


DRUG_CHEBI_ID = "CHEBI:23888"


def load_chebi_sdf(
    path: str | Path,
) -> list[dict[str, str | None]]:
    """Load raw ChEBI structure records from a compressed SDF."""
    path = Path(path)

    records: list[dict[str, str | None]] = []

    with gzip.open(path, "rb") as file:
        supplier = Chem.ForwardSDMolSupplier(
            file,
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )

        for molecule in supplier:
            if molecule is None:
                continue

            records.append(
                {
                    "chebi_id": _get_property(molecule, "ChEBI ID"),
                    "name": _get_property(molecule, "ChEBI NAME"),
                    "smiles": _get_property(molecule, "SMILES"),
                    "inchi": _get_property(molecule, "InChI"),
                    "inchikey": _get_property(
                        molecule,
                        "InChIKey",
                    ),
                }
            )

    return records


def _get_property(
    molecule: Chem.Mol,
    name: str,
) -> str | None:
    """Return an SDF property when present."""
    if molecule.HasProp(name):
        return molecule.GetProp(name)

    return None