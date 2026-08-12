from rdkit import Chem


def test_rdkit_smiles():
    smiles = "CCO"

    molecule = Chem.MolFromSmiles(smiles)

    assert molecule is not None