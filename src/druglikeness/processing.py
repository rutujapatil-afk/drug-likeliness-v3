from .records import MolecularRecord
from .standardization import canonicalize_smiles
from .validation import is_valid_smiles


def process_molecular_record(
    source_id: str,
    original_smiles: str,
) -> MolecularRecord:
    """
    Validate and standardize a molecular record.

    Parameters
    ----------
    source_id:
        Identifier associated with the source record.

    original_smiles:
        Original SMILES supplied by the source.

    Returns
    -------
    MolecularRecord
        Processed molecular record containing both original
        and standardized representations.
    """
    valid = is_valid_smiles(original_smiles)

    if not valid:
        return MolecularRecord(
            source_id=source_id,
            original_smiles=original_smiles,
            canonical_smiles=None,
            valid=False,
            standardization_status="invalid_smiles",
        )

    canonical_smiles = canonicalize_smiles(original_smiles)

    if canonical_smiles is None:
        return MolecularRecord(
            source_id=source_id,
            original_smiles=original_smiles,
            canonical_smiles=None,
            valid=False,
            standardization_status="standardization_failed",
        )

    return MolecularRecord(
        source_id=source_id,
        original_smiles=original_smiles,
        canonical_smiles=canonical_smiles,
        valid=True,
        standardization_status="standardized",
    )