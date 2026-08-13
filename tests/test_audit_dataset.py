import pandas as pd

from druglikeness.scaffold_split import (
    scaffold_from_smiles,
    scaffold_split,
)
from druglikeness.splitting import split_dataset


DATASET_PATH = "data/processed/modeling_dataset.csv"


def test_modeling_dataset_has_no_duplicate_molecules():
    dataframe = pd.read_csv(DATASET_PATH)

    assert dataframe["canonical_smiles"].is_unique


def test_modeling_dataset_has_no_cross_class_overlap():
    dataframe = pd.read_csv(DATASET_PATH)

    positive = set(
        dataframe.loc[
            dataframe["label"] == 1,
            "canonical_smiles",
        ]
    )

    negative = set(
        dataframe.loc[
            dataframe["label"] == 0,
            "canonical_smiles",
        ]
    )

    assert not positive & negative


def test_random_split_has_no_molecular_overlap():
    dataframe = pd.read_csv(DATASET_PATH)

    train, validation, test = split_dataset(dataframe)

    train_ids = set(train["canonical_smiles"])
    validation_ids = set(validation["canonical_smiles"])
    test_ids = set(test["canonical_smiles"])

    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_scaffold_split_has_no_scaffold_overlap():
    dataframe = pd.read_csv(DATASET_PATH)

    train, validation, test = scaffold_split(dataframe)

    train_scaffolds = {
        scaffold_from_smiles(smiles)
        for smiles in train["canonical_smiles"]
    }

    validation_scaffolds = {
        scaffold_from_smiles(smiles)
        for smiles in validation["canonical_smiles"]
    }

    test_scaffolds = {
        scaffold_from_smiles(smiles)
        for smiles in test["canonical_smiles"]
    }

    assert not train_scaffolds & validation_scaffolds
    assert not train_scaffolds & test_scaffolds
    assert not validation_scaffolds & test_scaffolds