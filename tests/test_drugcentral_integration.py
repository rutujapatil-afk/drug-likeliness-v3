from pathlib import Path

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


def test_drugcentral_full_dataset():
    dataframe = load_drugcentral(DATASET_PATH)

    records = process_drugcentral(dataframe)
    summary = summarize_drugcentral(records)

    assert summary["source"] == "DrugCentral"
    assert summary["records"] == 4099
    assert summary["valid_smiles"] == 4099
    assert summary["invalid_smiles"] == 0
    assert summary["canonical_smiles"] == 4099
    assert summary["unique_canonical_smiles"] == 4098
    assert summary["duplicate_canonical_smiles"] == 1