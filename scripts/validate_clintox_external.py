from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
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

EXTERNAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "clintox.csv.gz"
)

DEVELOPMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "clintox_external_predictions.csv"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "clintox_external_summary.csv"
)

RANDOM_SEED = 20260813
RADIUS = 2
N_BITS = 2048


def canonicalize(smiles):
    """Return canonical SMILES or None for invalid structures."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    # Reject wildcard/query atoms for this structure-based evaluation.
    if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
        return None

    return Chem.MolToSmiles(
        molecule,
        canonical=True,
    )


def make_fingerprints(smiles_list):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=N_BITS,
    )

    fingerprints = []

    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            raise ValueError(
                f"Invalid standardized SMILES: {smiles}"
            )

        fingerprints.append(
            generator.GetFingerprint(molecule)
        )

    return fingerprints


def fingerprints_to_array(fingerprints):
    arrays = []

    for fingerprint in fingerprints:
        array = np.zeros(
            N_BITS,
            dtype=np.uint8,
        )

        from rdkit.DataStructs import ConvertToNumpyArray

        ConvertToNumpyArray(
            fingerprint,
            array,
        )

        arrays.append(array)

    return np.vstack(arrays)


def main():
    print("=== ClinTox external validation ===")

    external = pd.read_csv(
        EXTERNAL_PATH,
        compression="gzip",
    )

    development = pd.read_csv(
        DEVELOPMENT_PATH
    )

    print("Raw external rows:", len(external))

    # Standardize external structures.
    external["canonical_smiles"] = (
        external["smiles"]
        .map(canonicalize)
    )

    invalid_count = int(
        external["canonical_smiles"].isna().sum()
    )

    external = external.dropna(
        subset=["canonical_smiles"]
    ).copy()

    print("Invalid/query structures excluded:", invalid_count)
    print("Valid external structures:", len(external))

    duplicate_count = int(
        external["canonical_smiles"].duplicated().sum()
    )

    external = external.drop_duplicates(
        subset=["canonical_smiles"]
    ).copy()

    print("Within-external duplicates excluded:", duplicate_count)

    development_smiles = set(
        development["canonical_smiles"]
    )

    overlap_mask = external[
        "canonical_smiles"
    ].isin(development_smiles)

    overlap_count = int(overlap_mask.sum())

    external = external.loc[
        ~overlap_mask
    ].copy()

    print(
        "Development-set overlap excluded:",
        overlap_count,
    )

    print(
        "Final external molecules:",
        len(external),
    )

    print("\nExternal labels:")
    print(
        external["FDA_APPROVED"]
        .value_counts()
        .sort_index()
    )

    if external["FDA_APPROVED"].nunique() != 2:
        raise ValueError(
            "External dataset does not contain both classes."
        )

    # Reconstruct the locked training split.
    train, validation, test = scaffold_split(
        development
    )

    print("\nLocked model training molecules:", len(train))
    print("Development validation molecules:", len(validation))
    print("Development test molecules:", len(test))

    train_fps = make_fingerprints(
        train["canonical_smiles"]
    )

    external_fps = make_fingerprints(
        external["canonical_smiles"]
    )

    X_train = fingerprints_to_array(
        train_fps
    )

    X_external = fingerprints_to_array(
        external_fps
    )

    y_train = train["label"].to_numpy()
    y_external = external[
        "FDA_APPROVED"
    ].to_numpy()

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_external
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = external[
        [
            "smiles",
            "canonical_smiles",
            "FDA_APPROVED",
            "CT_TOX",
        ]
    ].copy()

    result["predicted_probability"] = probabilities
    result["predicted_label"] = predictions
    result["correct"] = (
        result["FDA_APPROVED"]
        == result["predicted_label"]
    )

    metrics = {
        "roc_auc": roc_auc_score(
            y_external,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_external,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_external,
            predictions,
        ),
        "precision": precision_score(
            y_external,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_external,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_external,
            predictions,
            zero_division=0,
        ),
        "n": len(result),
        "positives": int(y_external.sum()),
        "negatives": int(
            len(y_external) - y_external.sum()
        ),
        "invalid_excluded": invalid_count,
        "duplicate_excluded": duplicate_count,
        "development_overlap_excluded": overlap_count,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n=== External metrics ===")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_external,
            predictions,
        )
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("\nSummary:")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()