from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier

from druglikeness.features import dataframe_to_morgan
from druglikeness.scaffold_split import scaffold_split


RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "scaffold_test_predictions.csv"
)

RANDOM_SEED = 20260813


def descriptors_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "heavy_atoms": Lipinski.HeavyAtomCount(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "logp": Crippen.MolLogP(mol),
    }


def main():
    dataframe = pd.read_csv(DATASET_PATH)

    train, validation, test = scaffold_split(dataframe)

    X_train = dataframe_to_morgan(train)
    X_test = dataframe_to_morgan(test)

    y_train = train["label"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = test.copy()

    result["predicted_probability"] = probabilities
    result["predicted_label"] = predictions
    result["correct"] = (
        result["label"] == result["predicted_label"]
    )

    descriptor_rows = [
        descriptors_from_smiles(smiles)
        for smiles in result["canonical_smiles"]
    ]

    descriptors = pd.DataFrame(descriptor_rows)

    result = pd.concat(
        [
            result.reset_index(drop=True),
            descriptors,
        ],
        axis=1,
    )

    result["error_type"] = "correct"

    result.loc[
        (result["label"] == 0)
        & (result["predicted_label"] == 1),
        "error_type",
    ] = "false_positive"

    result.loc[
        (result["label"] == 1)
        & (result["predicted_label"] == 0),
        "error_type",
    ] = "false_negative"

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Scaffold test error analysis ===")
    print("Test molecules:", len(result))

    print("\nError counts:")
    print(result["error_type"].value_counts())

    print("\nMean prediction probability by error type:")
    print(
        result.groupby("error_type")[
            "predicted_probability"
        ].mean().round(4)
    )

    descriptor_columns = [
        "molecular_weight",
        "heavy_atoms",
        "hbd",
        "hba",
        "rotatable_bonds",
        "rings",
        "aromatic_rings",
        "fraction_csp3",
        "logp",
    ]

    print("\nDescriptor means by error type:")
    print(
        result.groupby("error_type")[
            descriptor_columns
        ].mean().round(3)
    )

    print("\nFalse positives:")
    print(
        result.loc[
            result["error_type"] == "false_positive",
            [
                "source_id",
                "canonical_smiles",
                "predicted_probability",
            ],
        ]
        .sort_values(
            "predicted_probability",
            ascending=False,
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nFalse negatives:")
    print(
        result.loc[
            result["error_type"] == "false_negative",
            [
                "source_id",
                "canonical_smiles",
                "predicted_probability",
            ],
        ]
        .sort_values(
            "predicted_probability",
            ascending=True,
        )
        .head(20)
        .to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()