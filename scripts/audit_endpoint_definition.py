from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from druglikeness.features import dataframe_to_morgan


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)

CHARACTERIZATION_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "dataset_characterization.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "endpoint_definition_audit.csv"
)

RANDOM_SEED = 20260813


def main():
    print("=== Endpoint definition / source-label audit ===")

    df = pd.read_csv(DATASET_PATH)
    chars = pd.read_csv(CHARACTERIZATION_PATH)

    print(f"Dataset size: {len(df)}")

    # ---------------------------------------------------------
    # 1. Verify exact equivalence between label and source.
    # ---------------------------------------------------------

    expected_label = (
        df["source_dataset"]
        .map(
            {
                "ChEBI": 0,
                "DrugCentral": 1,
            }
        )
    )

    exact_equivalence = (
        df["label"].to_numpy()
        == expected_label.to_numpy()
    )

    mismatches = int((~exact_equivalence).sum())

    print("\n=== Label/source equivalence ===")
    print("Label/source mismatches:", mismatches)

    if mismatches == 0:
        print(
            "WARNING: label is exactly determined by source_dataset."
        )
    else:
        print(
            "Label is not perfectly determined by source_dataset."
        )

    # ---------------------------------------------------------
    # 2. Dataset composition.
    # ---------------------------------------------------------

    composition = (
        pd.crosstab(
            df["source_dataset"],
            df["label"],
        )
    )

    print("\n=== Source × label ===")
    print(composition)

    # ---------------------------------------------------------
    # 3. Descriptor source differences.
    # ---------------------------------------------------------

    merged = chars.merge(
        df[
            [
                "canonical_smiles",
                "source_dataset",
            ]
        ],
        on="canonical_smiles",
        how="inner",
    )

    descriptors = [
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

    rows = []

    for descriptor in descriptors:
        for source in ["ChEBI", "DrugCentral"]:
            values = merged.loc[
                merged["source_dataset"] == source,
                descriptor,
            ].dropna()

            rows.append(
                {
                    "analysis": "descriptor_source_distribution",
                    "descriptor": descriptor,
                    "source": source,
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "sd": values.std(ddof=1),
                }
            )

    # ---------------------------------------------------------
    # 4. Can Morgan fingerprints recover source identity?
    # ---------------------------------------------------------

    print("\n=== Morgan fingerprint source classifier ===")

    X = dataframe_to_morgan(df)
    y = (
        df["source_dataset"]
        .eq("DrugCentral")
        .astype(int)
        .to_numpy()
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    source_roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    source_accuracy = accuracy_score(
        y,
        predictions,
    )

    print(
        f"5-fold source-classification ROC-AUC: "
        f"{source_roc_auc:.4f}"
    )

    print(
        f"5-fold source-classification accuracy: "
        f"{source_accuracy:.4f}"
    )

    rows.append(
        {
            "analysis": "Morgan_source_classifier",
            "descriptor": "Morgan fingerprints",
            "source": "DrugCentral_vs_ChEBI",
            "n": len(df),
            "mean": source_roc_auc,
            "median": source_accuracy,
            "sd": np.nan,
        }
    )

    # ---------------------------------------------------------
    # 5. Save audit.
    # ---------------------------------------------------------

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== Interpretation ===")

    if mismatches == 0:
        print(
            "The current binary endpoint is exactly equivalent "
            "to source membership."
        )

    print(
        "The Morgan source-classification result quantifies "
        "how strongly molecular representation encodes source identity."
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()