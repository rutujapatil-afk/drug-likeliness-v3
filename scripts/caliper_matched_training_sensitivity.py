from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import druglikeness.scaffold_split as scaffold_module
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

CALIPER = 0.5

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

    result[DESCRIPTORS] = pd.DataFrame(
        values,
        columns=DESCRIPTORS,
        index=result.index,
    )

    return result


def smd(a, b):
    pooled_sd = np.sqrt(
        (
            a.var(ddof=1)
            + b.var(ddof=1)
        )
        / 2
    )

    if pooled_sd == 0:
        return 0.0

    return (
        (a.mean() - b.mean())
        / pooled_sd
    )


def caliper_match(
    positives,
    negatives,
    caliper,
):
    """
    Greedy one-to-one caliper matching.

    Matching is performed only within the supplied training split.

    Candidates are processed in order of the number of available
    neighbors, so structurally constrained molecules are matched first.
    """

    scaler = StandardScaler()

    positive_x = scaler.fit_transform(
        positives[DESCRIPTORS]
    )

    negative_x = scaler.transform(
        negatives[DESCRIPTORS]
    )

    # Find all negative candidates within the caliper.
    neighbor_model = NearestNeighbors(
        radius=caliper,
        metric="euclidean",
        algorithm="auto",
    )

    neighbor_model.fit(negative_x)

    distances, indices = neighbor_model.radius_neighbors(
        positive_x,
        return_distance=True,
    )

    candidate_counts = np.array(
        [
            len(candidate_indices)
            for candidate_indices in indices
        ]
    )

    # Hardest-to-match positives first.
    order = np.argsort(candidate_counts)

    used_negative_positions = set()

    matched_positive_positions = []
    matched_negative_positions = []
    matched_distances = []

    for positive_position in order:

        candidate_indices = indices[
            positive_position
        ]

        candidate_distances = distances[
            positive_position
        ]

        available = [
            (distance, negative_position)
            for distance, negative_position
            in zip(
                candidate_distances,
                candidate_indices,
            )
            if negative_position
            not in used_negative_positions
        ]

        if not available:
            continue

        distance, negative_position = min(
            available,
            key=lambda item: item[0],
        )

        used_negative_positions.add(
            int(negative_position)
        )

        matched_positive_positions.append(
            int(positive_position)
        )

        matched_negative_positions.append(
            int(negative_position)
        )

        matched_distances.append(
            float(distance)
        )

    matched_positive = positives.iloc[
        matched_positive_positions
    ].copy()

    matched_negative = negatives.iloc[
        matched_negative_positions
    ].copy()

    return (
        matched_positive,
        matched_negative,
        matched_distances,
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

    results = []

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

        results.append(
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

    return results


def main():
    dataframe = pd.read_csv(
        DATASET_PATH
    )

    dataframe = add_descriptors(
        dataframe
    )

    all_results = []
    all_balance = []

    for seed in SEEDS:

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

        matched_positive, matched_negative, distances = (
            caliper_match(
                train_positive,
                train_negative,
                CALIPER,
            )
        )

        matched_train = pd.concat(
            [
                matched_positive,
                matched_negative,
            ],
            ignore_index=True,
        )

        for descriptor in DESCRIPTORS:
            all_balance.append(
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
            all_results.append(
                {
                    "seed": seed,
                    "caliper": CALIPER,
                    "original_train_size": len(train),
                    "matched_train_size": len(matched_train),
                    "matched_pairs": len(matched_positive),
                    "match_fraction_positive": (
                        len(matched_positive)
                        / len(train_positive)
                    ),
                    "median_match_distance": np.median(
                        distances
                    ),
                    "mean_match_distance": np.mean(
                        distances
                    ),
                    **metric,
                }
            )

        print(
            f"Seed {seed}: "
            f"{len(matched_positive)} matched pairs "
            f"from {len(train_positive)} positives"
        )

        print(
            f"  Match fraction: "
            f"{len(matched_positive) / len(train_positive):.3f}"
        )

        print(
            f"  Median distance: "
            f"{np.median(distances):.3f}"
        )

        for metric in metrics:
            print(
                f"  {metric['split']}: "
                f"ROC-AUC={metric['roc_auc']:.4f}, "
                f"PR-AUC={metric['pr_auc']:.4f}"
            )

    results = pd.DataFrame(
        all_results
    )

    balance = pd.DataFrame(
        all_balance
    )

    results_path = (
        OUTPUT_DIR
        / "caliper_matched_training_results.csv"
    )

    balance_path = (
        OUTPUT_DIR
        / "caliper_matched_descriptor_balance.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    balance.to_csv(
        balance_path,
        index=False,
    )

    print("\n=== Summary ===")

    for split_name in [
        "validation",
        "test",
    ]:

        subset = results[
            results["split"] == split_name
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

    print("\n=== Matching ===")

    print(
        results[
            [
                "matched_pairs",
                "match_fraction_positive",
                "median_match_distance",
                "mean_match_distance",
            ]
        ].drop_duplicates().to_string(
            index=False
        )
    )

    print(
        "\n=== Descriptor balance ==="
    )

    balance_summary = (
        balance
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