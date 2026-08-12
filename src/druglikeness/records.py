from dataclasses import dataclass


@dataclass(frozen=True)
class MolecularRecord:
    """
    Represents a molecular record during preprocessing.

    Attributes
    ----------
    source_id:
        Identifier associated with the source record.

    original_smiles:
        Original SMILES exactly as supplied by the source.

    canonical_smiles:
        Canonical SMILES generated during standardization.
        None when standardization was unsuccessful.

    valid:
        Whether the original SMILES could be parsed by RDKit.

    standardization_status:
        Processing status for the molecular record.
    """

    source_id: str
    original_smiles: str
    canonical_smiles: str | None
    valid: bool
    standardization_status: str