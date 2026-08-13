from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

SEEDS = [
    20260813,
    20260814,
    20260815,
    20260816,
    20260817,
]

DESCRIPTORS = [
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


def add_descriptors(dataframe):
    result = dataframe.copy()

    values = [
        descriptors_from_smiles(smiles)
        for smiles in result["canonical_smiles"]
    ]

    descriptor_frame = pd.DataFrame(
        values,
        columns=DESCRIPTORS,
        index=result.index,
    )

    return pd.concat(
        [
            result,
            descriptor_frame,
        ],
        axis=1,
    )


def smd(a, b):
    mean_a = a.mean()
    mean_b = b.mean()

    pooled_sd = np.sqrt(
        (a.var(ddof=1) + b.var(ddof=1)) / 2
    )

    if pooled_sd == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_sd


def match_training_negatives(
    positives,
    negatives,
):
    """
    Match the smaller training class to the larger one.

    Matching is performed ONLY within the training split.
    """

    if len(positives) <= len(negatives):
        smaller = positives
        larger = negatives
        smaller_is_positive = True
    else:
        smaller = negatives
        larger = positives
        smaller_is_positive = False

    scaler = StandardScaler()

    smaller_matrix = scaler.fit_transform(
        smaller[DESCRIPTORS]
    )

    larger_matrix = scaler.transform(
        larger[DESCRIPTORS]
    )

    distances = (
        (
            smaller_matrix[:, None, :]
            - larger_matrix[None, :, :]
        )
        ** 2
    ).sum(axis=2)

    smaller_indices, larger_indices = (
        linear_sum_assignment(distances)
    )

    selected_larger = larger.iloc[
        larger_indices
    ].copy()

    if smaller_is_positive:
        matched_positive = smaller.copy()
        matched_negative = selected_larger
    else:
        matched_negative = smaller.copy()
        matched_positive = selected_larger

    return (
        matched_positive,
        matched_negative,
    )


def evaluate(
    train,
    validation,
    test,
    seed,
):
    X_train = dataframe_to_morgan(train)
    X_validation = dataframe_to_morgan(validation)
    X_test = dataframe_to_morgan(test)

    y_train = train["label"].to_numpy()
    y_validation = validation["label"].to_numpy()
    y_test = test["label"].to_numpy()

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train,
    )

    rows = []

    for split_name, X, y in [
        (
            "validation",
            X_validation,
            y_validation,
        ),
        (
            "test",
            X_test,
            y_test,
        ),
    ]:
        probabilities = model.predict_proba(
            X
        )[:, 1]

        rows.append(
            {
                "split": split_name,
                "roc_auc": roc_auc_score(
                    y,
                    probabilities,
                ),
                "pr_auc": average_precision_score(
                    y,
                    probabilities,
                ),
            }
        )

    return rows


def main():
    dataframe = pd.read_csv(
        DATASET_PATH
    )

    dataframe = add_descriptors(
        dataframe
    )

    results = []
    balance_results = []

    for seed in SEEDS:
        # The existing scaffold_split function is deterministic.
        # The repeated experiment therefore needs the split seed
        # to be temporarily changed.
        import druglikeness.scaffold_split as scaffold_module

        scaffold_module.RANDOM_SEED = seed

        train, validation, test = scaffold_split(
            dataframe
        )

        train_positive = train[
            train["label"] == 1
        ].copy()

        train_negative = train[
            train["label"] == 0
        ].copy()

        matched_positive, matched_negative = (
            match_training_negatives(
                train_positive,
                train_negative,
            )
        )

        matched_train = pd.concat(
            [
                matched_positive,
                matched_negative,
            ],
            ignore_index=True,
        )

        # Descriptor balance is evaluated ONLY in training data.
        for descriptor in DESCRIPTORS:
            balance_results.append(
                {
                    "seed": seed,
                    "descriptor": descriptor,
                    "smd_before": smd(
                        train_positive[descriptor],
                        train_negative[descriptor],
                    ),
                    "smd_after": smd(
                        matched_positive[descriptor],
                        matched_negative[descriptor],
                    ),
                }
            )

        metrics = evaluate(
            matched_train,
            validation,
            test,
            seed,
        )

        for metric in metrics:
            metric["seed"] = seed
            metric["matched_train_size"] = len(
                matched_train
            )
            metric["original_train_size"] = len(
                train
            )

            results.append(metric)

        print(
            f"Seed {seed}: "
            f"train {len(train)} -> "
            f"matched {len(matched_train)}"
        )

        for metric in metrics:
            print(
                f"  {metric['split']}: "
                f"ROC-AUC={metric['roc_auc']:.4f}, "
                f"PR-AUC={metric['pr_auc']:.4f}"
            )

    results_frame = pd.DataFrame(
        results
    )

    balance_frame = pd.DataFrame(
        balance_results
    )

    results_path = (
        OUTPUT_DIR
        / "matched_training_sensitivity.csv"
    )

    balance_path = (
        OUTPUT_DIR
        / "matched_training_descriptor_balance.csv"
    )

    results_frame.to_csv(
        results_path,
        index=False,
    )

    balance_frame.to_csv(
        balance_path,
        index=False,
    )

    print("\n=== Summary ===")

    for split_name in [
        "validation",
        "test",
    ]:
        subset = results_frame[
            results_frame["split"] == split_name
        ]

        print(
            f"\n{split_name}:"
        )

        for metric in [
            "roc_auc",
            "pr_auc",
        ]:
            print(
                f"{metric}: "
                f"{subset[metric].mean():.4f} +/- "
                f"{subset[metric].std(ddof=1):.4f}"
            )

    print("\n=== Descriptor balance ===")

    balance_summary = (
        balance_frame
        .groupby("descriptor")[
            [
                "smd_before",
                "smd_after",
            ]
        ]
        .mean()
    )

    print(
        balance_summary.round(3)
    )

    print("\nOutputs:")
    print(results_path)
    print(balance_path)


if __name__ == "__main__":
    main()