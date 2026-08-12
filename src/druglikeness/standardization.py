from rdkit import Chem


def canonicalize_smiles(smiles: str) -> str | None:
    """
    Convert a valid SMILES string to RDKit's canonical SMILES.

    Parameters
    ----------
    smiles:
        Molecular structure represented as a SMILES string.

    Returns
    -------
    str | None
        Canonical SMILES if the input is valid, otherwise None.
    """
    if not isinstance(smiles, str):
        return None

    smiles = smiles.strip()

    if not smiles:
        return None

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    return Chem.MolToSmiles(molecule, canonical=True)