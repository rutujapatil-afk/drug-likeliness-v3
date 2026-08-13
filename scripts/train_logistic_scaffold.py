from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

RANDOM_SEED = 20260813


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

    # IMPORTANT:
    # Use the scaffold split, not the random split.
    train, validation, test = scaffold_split(dataframe)

    X_train = dataframe_to_morgan(train)
    X_validation = dataframe_to_morgan(validation)
    X_test = dataframe_to_morgan(test)

    y_train = train["label"].to_numpy()
    y_validation = validation["label"].to_numpy()
    y_test = test["label"].to_numpy()

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        solver="liblinear",
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

    print("=== Logistic Regression Scaffold Validation ===")
    for key, value in validation_metrics.items():
        print(f"{key}: {value}")

    print()
    print("=== Logistic Regression Scaffold Test ===")
    for key, value in test_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()