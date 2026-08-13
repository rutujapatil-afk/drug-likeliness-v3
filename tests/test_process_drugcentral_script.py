from pathlib import Path

import pandas as pd

from druglikeness.drugcentral import (
    load_drugcentral,
    process_drugcentral,
    summarize_drugcentral,
)


DATASET_PATH = (
    Path(__file__).parents[1]
    / "data"
    / "raw"
    / "structures.smiles.tsv"
)


def test_drugcentral_processed_output():
    dataframe = load_drugcentral(DATASET_PATH)
    records = process_drugcentral(dataframe)

    processed = pd.DataFrame(
        [
            {
                "source_id": record.source_id,
                "original_smiles": record.original_smiles,
                "canonical_smiles": record.canonical_smiles,
                "valid": record.valid,
                "standardization_status": (
                    record.standardization_status
                ),
            }
            for record in records
        ]
    )

    summary = summarize_drugcentral(records)

    assert processed.shape == (4099, 5)
    assert processed["valid"].sum() == 4099
    assert processed["canonical_smiles"].notna().sum() == 4099
    assert processed["canonical_smiles"].nunique() == 4098

    assert summary["duplicate_canonical_smiles"] == 1