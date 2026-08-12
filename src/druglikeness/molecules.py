from rdkit import Chem


def smiles_to_molecule(smiles: str):
    """
    Convert a SMILES string into an RDKit molecule.

    Parameters
    ----------
    smiles:
        Molecular structure represented as a SMILES string.

    Returns
    -------
    rdkit.Chem.Mol | None
        RDKit molecule if the SMILES is valid, otherwise None.
    """
    if not isinstance(smiles, str):
        raise TypeError("SMILES must be provided as a string.")

    smiles = smiles.strip()

    if not smiles:
        return None

    return Chem.MolFromSmiles(smiles)