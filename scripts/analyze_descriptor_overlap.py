from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


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
    / "descriptor_overlap_analysis.csv"
)

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


def main():
    dataframe = pd.read_csv(DATASET_PATH)

    descriptor_values = np.asarray(
        [
            descriptors_from_smiles(smiles)
            for smiles in dataframe["canonical_smiles"]
        ]
    )

    X = StandardScaler().fit_transform(
        descriptor_values
    )

    positive_mask = (
        dataframe["label"].to_numpy() == 1
    )

    negative_mask = ~positive_mask

    positive_X = X[positive_mask]
    negative_X = X[negative_mask]

    # Nearest ChEBI molecule for every DrugCentral molecule.
    nn_negative = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )

    nn_negative.fit(negative_X)

    positive_distances, _ = nn_negative.kneighbors(
        positive_X
    )

    # Nearest DrugCentral molecule for every ChEBI molecule.
    nn_positive = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )

    nn_positive.fit(positive_X)

    negative_distances, _ = nn_positive.kneighbors(
        negative_X
    )

    positive_distances = positive_distances[:, 0]
    negative_distances = negative_distances[:, 0]

    rows = []

    for threshold in [
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
    ]:
        rows.append(
            {
                "class": "DrugCentral",
                "threshold": threshold,
                "n_total": len(
                    positive_distances
                ),
                "n_within_threshold": int(
                    (
                        positive_distances
                        <= threshold
                    ).sum()
                ),
                "fraction_within_threshold": (
                    positive_distances
                    <= threshold
                ).mean(),
                "median_nearest_distance":
                    np.median(
                        positive_distances
                    ),
                "mean_nearest_distance":
                    np.mean(
                        positive_distances
                    ),
            }
        )

        rows.append(
            {
                "class": "ChEBI",
                "threshold": threshold,
                "n_total": len(
                    negative_distances
                ),
                "n_within_threshold": int(
                    (
                        negative_distances
                        <= threshold
                    ).sum()
                ),
                "fraction_within_threshold": (
                    negative_distances
                    <= threshold
                ).mean(),
                "median_nearest_distance":
                    np.median(
                        negative_distances
                    ),
                "mean_nearest_distance":
                    np.mean(
                        negative_distances
                    ),
            }
        )

    result = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Descriptor common-support analysis ===")

    print(
        "\nDrugCentral → nearest ChEBI:"
    )
    print(
        f"median distance: "
        f"{np.median(positive_distances):.3f}"
    )
    print(
        f"mean distance: "
        f"{np.mean(positive_distances):.3f}"
    )

    print(
        "\nChEBI → nearest DrugCentral:"
    )
    print(
        f"median distance: "
        f"{np.median(negative_distances):.3f}"
    )
    print(
        f"mean distance: "
        f"{np.mean(negative_distances):.3f}"
    )

    print("\nCommon support:")
    print(
        result.to_string(index=False)
    )

    print("\nOutput:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()