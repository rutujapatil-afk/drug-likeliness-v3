import pandas as pd
import pytest

from druglikeness.overlap import (
    find_conflicting_labels,
    find_cross_dataset_overlap,
    find_duplicate_molecules,
)


def test_find_duplicate_molecules():
    dataframe = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCO",
                "CC(=O)O",
                "C1CCCCC1",
            ]
        }
    )

    duplicates = find_duplicate_molecules(dataframe)

    assert len(duplicates) == 2
    assert set(duplicates["canonical_smiles"]) == {"CCO"}


def test_find_duplicate_molecules_ignores_missing():
    dataframe = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                None,
                None,
            ]
        }
    )

    duplicates = find_duplicate_molecules(dataframe)

    assert len(duplicates) == 0


def test_find_cross_dataset_overlap():
    first = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CC(=O)O",
                "C1CCCCC1",
            ]
        }
    )

    second = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "C1CCCCC1",
                "CCCC",
            ]
        }
    )

    overlap = find_cross_dataset_overlap(
        first,
        second,
    )

    assert set(overlap["canonical_smiles"]) == {
        "CCO",
        "C1CCCCC1",
    }


def test_find_conflicting_labels():
    dataframe = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCO",
                "CC(=O)O",
            ],
            "label": [
                1,
                0,
                1,
            ],
        }
    )

    conflicts = find_conflicting_labels(dataframe)

    assert len(conflicts) == 2
    assert set(conflicts["canonical_smiles"]) == {"CCO"}


def test_no_conflicting_labels():
    dataframe = pd.DataFrame(
        {
            "canonical_smiles": [
                "CCO",
                "CCO",
                "CC(=O)O",
            ],
            "label": [
                1,
                1,
                0,
            ],
        }
    )

    conflicts = find_conflicting_labels(dataframe)

    assert conflicts.empty


def test_overlap_requires_canonical_smiles():
    dataframe = pd.DataFrame(
        {
            "smiles": ["CCO"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        find_duplicate_molecules(dataframe)