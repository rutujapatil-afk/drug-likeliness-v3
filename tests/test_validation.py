from druglikeness.validation import is_valid_smiles


def test_valid_smiles():
    assert is_valid_smiles("CCO") is True


def test_invalid_smiles():
    assert is_valid_smiles("this-is-not-a-smiles") is False


def test_empty_smiles():
    assert is_valid_smiles("") is False


def test_whitespace_smiles():
    assert is_valid_smiles("   ") is False


def test_non_string_smiles():
    assert is_valid_smiles(None) is False