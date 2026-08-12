from druglikeness.standardization import canonicalize_smiles


def test_canonicalize_smiles():
    result = canonicalize_smiles("C(C)O")

    assert result == "CCO"


def test_canonicalize_smiles_equivalent_representation():
    result_a = canonicalize_smiles("C(C)O")
    result_b = canonicalize_smiles("OCC")

    assert result_a == result_b


def test_canonicalize_invalid_smiles():
    result = canonicalize_smiles("this-is-not-a-smiles")

    assert result is None


def test_canonicalize_empty_smiles():
    result = canonicalize_smiles("")

    assert result is None