from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from druglikeness.scaffold_split import scaffold_split


RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "modeling_dataset.csv"
)

RANDOM_SEED = 20260813


def descriptors_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return [
        Descriptors.MolWt(mol),
        Lipinski.HeavyAtomCount(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Crippen.MolLogP(mol),
    ]


def make_features(dataframe):
    return [
        descriptors_from_smiles(smiles)
        for smiles in dataframe["canonical_smiles"]
    ]


def evaluate(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1": f1_score(y, predictions),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
    }


def main():
    dataframe = pd.read_csv(DATASET_PATH)

    # Corrected scaffold-separated split
    train, validation, test = scaffold_split(dataframe)

    X_train = make_features(train)
    X_validation = make_features(validation)
    X_test = make_features(test)

    y_train = train["label"].to_numpy()
    y_validation = validation["label"].to_numpy()
    y_test = test["label"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
    )

    print("=== Descriptor RF Scaffold Validation ===")
    for key, value in validation_metrics.items():
        print(f"{key}: {value}")

    print()
    print("=== Descriptor RF Scaffold Test ===")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()