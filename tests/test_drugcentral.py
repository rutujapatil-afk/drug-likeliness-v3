from pathlib import Path

import pandas as pd

from druglikeness.drugcentral import (
    EXPECTED_COLUMNS,
    load_drugcentral,
    process_drugcentral,
)


FIXTURE_PATH = Path(
    "tests/fixtures/drugcentral_sample.tsv"
)


def test_expected_columns():
    assert EXPECTED_COLUMNS == [
        "SMILES",
        "InChI",
        "InChIKey",
        "ID",
        "INN",
        "CAS_RN",
    ]


def test_load_drugcentral():
    dataframe = load_drugcentral(FIXTURE_PATH)

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == EXPECTED_COLUMNS
    assert len(dataframe) == 3


def test_process_drugcentral():
    dataframe = load_drugcentral(FIXTURE_PATH)

    records = process_drugcentral(dataframe)

    assert len(records) == 3

    assert records[0].source_id == "5392"
    assert records[0].original_smiles == "CCO"
    assert records[0].canonical_smiles == "CCO"
    assert records[0].valid is True
    assert records[0].standardization_status == "standardized"


def test_invalid_smiles_is_retained():
    dataframe = load_drugcentral(FIXTURE_PATH)

    records = process_drugcentral(dataframe)

    invalid_record = records[2]

    assert invalid_record.source_id == "5394"
    assert invalid_record.original_smiles == "this-is-invalid"
    assert invalid_record.canonical_smiles is None
    assert invalid_record.valid is False
    assert invalid_record.standardization_status == "invalid_smiles"