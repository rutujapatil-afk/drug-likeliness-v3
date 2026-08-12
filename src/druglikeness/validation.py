from rdkit import Chem


def is_valid_smiles(smiles: str) -> bool:
    """
    Determine whether a SMILES string can be parsed by RDKit.

    Parameters
    ----------
    smiles:
        Molecular structure represented as a SMILES string.

    Returns
    -------
    bool
        True if the SMILES is valid, otherwise False.
    """
    if not isinstance(smiles, str):
        return False

    smiles = smiles.strip()

    if not smiles:
        return False

    molecule = Chem.MolFromSmiles(smiles)

    return molecule is not None