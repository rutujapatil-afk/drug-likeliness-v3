from pathlib import Path

import pandas as pd
import pytest

from druglikeness.dataset import (
    load_molecular_csv,
    process_molecular_dataframe,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "test_molecules.csv"
)


def test_load_molecular_csv():
    dataframe = load_molecular_csv(FIXTURE_PATH)

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == ["source_id", "smiles"]
    assert len(dataframe) == 6


def test_load_molecular_csv_requires_columns(tmp_path):
    invalid_file = tmp_path / "invalid.csv"

    invalid_file.write_text(
        "name,value\n"
        "example,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        load_molecular_csv(invalid_file)


def test_process_molecular_dataframe():
    dataframe = load_molecular_csv(FIXTURE_PATH)

    processed = process_molecular_dataframe(dataframe)

    assert len(processed) == 6

    assert list(processed.columns) == [
        "source_id",
        "original_smiles",
        "canonical_smiles",
        "valid",
        "standardization_status",
    ]

    assert processed.loc[0, "canonical_smiles"] == "CCO"
    assert bool(processed.loc[0, "valid"]) is True

    assert bool(processed.loc[3, "valid"]) is False
    assert processed.loc[3, "standardization_status"] == "invalid_smiles"