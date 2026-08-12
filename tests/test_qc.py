from pathlib import Path

import pandas as pd
import pytest

from druglikeness.dataset import (
    load_molecular_csv,
    process_molecular_dataframe,
)
from druglikeness.qc import (
    DatasetQCReport,
    generate_qc_report,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "test_molecules.csv"
)


@pytest.fixture
def processed_dataset() -> pd.DataFrame:
    dataframe = load_molecular_csv(FIXTURE_PATH)

    return process_molecular_dataframe(dataframe)


def test_generate_qc_report(processed_dataset):
    report = generate_qc_report(processed_dataset)

    assert isinstance(report, DatasetQCReport)

    assert report.total_records == 6
    assert report.valid_records == 5
    assert report.invalid_records == 1
    assert report.missing_smiles == 0
    assert report.duplicate_canonical_smiles == 2

    assert report.status_counts == {
        "standardized": 5,
        "invalid_smiles": 1,
    }


def test_qc_requires_columns():
    dataframe = pd.DataFrame(
        {
            "smiles": ["CCO"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_qc_report(dataframe)