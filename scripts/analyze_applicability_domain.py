from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

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
    / "applicability_domain.csv"
)

RANDOM_SEED = 20260813
RADIUS = 2
N_BITS = 2048


def make_generator():
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
    )


def make_fingerprints(dataframe):
    generator = make_generator()

    fingerprints = []

    for smiles in dataframe["canonical_smiles"]:
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fingerprints.append(
            generator.GetFingerprint(molecule)
        )

    return fingerprints


def nearest_training_similarity(test_fp, train_fps):
    similarities = DataStructs.BulkTanimotoSimilarity(
        test_fp,
        train_fps,
    )

    return max(similarities)


def main():
    dataframe = pd.read_csv(DATASET_PATH)

    train, validation, test = scaffold_split(dataframe)

    train_fps = make_fingerprints(train)
    test_fps = make_fingerprints(test)

    print("Training molecules:", len(train))
    print("Test molecules:", len(test))

    similarities = [
        nearest_training_similarity(
            test_fp,
            train_fps,
        )
        for test_fp in test_fps
    ]

    # Train the locked baseline model.
    X_train = np.asarray(
        [
            np.asarray(fp)
            for fp in train_fps
        ]
    )

    X_test = np.asarray(
        [
            np.asarray(fp)
            for fp in test_fps
        ]
    )

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    result = test[
        [
            "source_id",
            "canonical_smiles",
            "label",
        ]
    ].copy()

    result["max_train_tanimoto"] = similarities
    result["predicted_probability"] = probabilities

    bins = [
        0.0,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.000001,
    ]

    labels = [
        "<0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        ">=0.9",
    ]

    result["similarity_bin"] = pd.cut(
        result["max_train_tanimoto"],
        bins=bins,
        labels=labels,
        right=False,
    )

    summary_rows = []

    for similarity_bin, group in result.groupby(
        "similarity_bin",
        observed=False,
    ):
        if len(group) < 10:
            continue

        y = group["label"].to_numpy()
        p = group["predicted_probability"].to_numpy()

        summary_rows.append(
            {
                "similarity_bin": str(similarity_bin),
                "n": len(group),
                "positives": int(y.sum()),
                "negatives": int(len(y) - y.sum()),
                "mean_similarity": group[
                    "max_train_tanimoto"
                ].mean(),
                "roc_auc": (
                    roc_auc_score(y, p)
                    if len(np.unique(y)) == 2
                    else np.nan
                ),
                "pr_auc": (
                    average_precision_score(y, p)
                    if len(np.unique(y)) == 2
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary_path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "applicability_domain_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n=== Applicability-domain summary ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nMolecule-level output:")
    print(OUTPUT_PATH)

    print("\nSummary output:")
    print(summary_path)


if __name__ == "__main__":
    main()