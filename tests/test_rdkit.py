from rdkit import Chem

from druglikeness.molecules import smiles_to_molecule


def test_smiles_to_molecule_valid():
    molecule = smiles_to_molecule("CCO")

    assert molecule is not None
    assert isinstance(molecule, Chem.Mol)


def test_smiles_to_molecule_empty():
    molecule = smiles_to_molecule("")

    assert molecule is None


def test_smiles_to_molecule_whitespace():
    molecule = smiles_to_molecule("   ")

    assert molecule is None