from druglikeness.processing import process_molecular_record
from druglikeness.records import MolecularRecord


def test_valid_molecular_record():
    record = process_molecular_record(
        source_id="mol-001",
        original_smiles="OCC",
    )

    assert isinstance(record, MolecularRecord)
    assert record.source_id == "mol-001"
    assert record.original_smiles == "OCC"
    assert record.canonical_smiles == "CCO"
    assert record.valid is True
    assert record.standardization_status == "standardized"


def test_invalid_molecular_record():
    record = process_molecular_record(
        source_id="mol-002",
        original_smiles="this-is-not-a-smiles",
    )

    assert record.source_id == "mol-002"
    assert record.original_smiles == "this-is-not-a-smiles"
    assert record.canonical_smiles is None
    assert record.valid is False
    assert record.standardization_status == "invalid_smiles"


def test_original_smiles_is_preserved():
    original = "OCC"

    record = process_molecular_record(
        source_id="mol-003",
        original_smiles=original,
    )

    assert record.original_smiles == original
    assert record.canonical_smiles == "CCO"