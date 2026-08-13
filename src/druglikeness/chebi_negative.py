from __future__ import annotations

import gzip
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger

from .chebi_ontology import load_chebi_drug_role_ids
from .standardization import canonicalize_smiles


RDLogger.DisableLog("rdApp.*")


def build_chebi_negative_candidates(
    sdf_path: str | Path,
    ontology_path: str | Path,
    excluded_smiles: set[str],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Build ChEBI negative-reference candidates with QC counts."""
    drug_role_ids = load_chebi_drug_role_ids(ontology_path)

    candidates: list[dict[str, object]] = []

    stats = {
        "total_records": 0,
        "unreadable_records": 0,
        "missing_smiles": 0,
        "drug_role_excluded": 0,
        "invalid_smiles": 0,
        "drugcentral_overlap_excluded": 0,
        "negative_candidates": 0,
    }

    with gzip.open(sdf_path, "rb") as file:
        supplier = Chem.ForwardSDMolSupplier(
            file,
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )

        for molecule in supplier:
            stats["total_records"] += 1

            if molecule is None:
                stats["unreadable_records"] += 1
                continue

            if not molecule.HasProp("ChEBI ID"):
                stats["unreadable_records"] += 1
                continue

            chebi_id = molecule.GetProp("ChEBI ID")

            if chebi_id in drug_role_ids:
                stats["drug_role_excluded"] += 1
                continue

            if not molecule.HasProp("SMILES"):
                stats["missing_smiles"] += 1
                continue

            smiles = molecule.GetProp("SMILES")

            try:
                canonical_smiles = canonicalize_smiles(smiles)
            except Exception:
                canonical_smiles = None

            if canonical_smiles is None:
                stats["invalid_smiles"] += 1
                continue

            if canonical_smiles in excluded_smiles:
                stats["drugcentral_overlap_excluded"] += 1
                continue

            candidates.append(
                {
                    "source_id": chebi_id,
                    "original_smiles": smiles,
                    "canonical_smiles": canonical_smiles,
                    "valid": True,
                    "standardization_status": "standardized",
                    "label": 0,
                }
            )

    stats["negative_candidates"] = len(candidates)

    return candidates, stats