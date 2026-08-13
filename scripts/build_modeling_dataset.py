from pathlib import Path
import random

import pandas as pd

from druglikeness.drugcentral import (
    load_drugcentral,
    process_drugcentral,
)
from druglikeness.chebi_negative import (
    build_chebi_negative_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRUGCENTRAL_PATH = (
    PROJECT_ROOT / "data" / "raw" / "structures.smiles.tsv"
)

CHEBI_SDF_PATH = (
    PROJECT_ROOT / "data" / "raw" / "chebi" / "chebi.sdf.gz"
)

CHEBI_ONTOLOGY_PATH = (
    PROJECT_ROOT / "data" / "raw" / "chebi" / "chebi.obo.gz"
)

NEGATIVE_POOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "chebi_negative_candidates.csv"
)

MODELING_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)

RANDOM_SEED = 20260813


def records_to_dataframe(records) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_id": record.source_id,
                "original_smiles": record.original_smiles,
                "canonical_smiles": record.canonical_smiles,
                "valid": record.valid,
                "standardization_status": (
                    record.standardization_status
                ),
            }
            for record in records
        ]
    )


def main() -> None:
    # -----------------------------
    # Positive DrugCentral dataset
    # -----------------------------
    drugcentral = load_drugcentral(DRUGCENTRAL_PATH)
    positive_records = process_drugcentral(drugcentral)

    positive = records_to_dataframe(
        positive_records
    )

    positive = positive[
        positive["valid"]
        & positive["canonical_smiles"].notna()
    ].drop_duplicates(
        subset=["canonical_smiles"]
    )

    positive["label"] = 1
    positive["source_dataset"] = "DrugCentral"

    # -----------------------------
    # ChEBI negative candidates
    # -----------------------------
    excluded_smiles = set(
        positive["canonical_smiles"]
    )

    candidates, stats = build_chebi_negative_candidates(
        CHEBI_SDF_PATH,
        CHEBI_ONTOLOGY_PATH,
        excluded_smiles,
    )

    negative_pool = pd.DataFrame(candidates)

    negative_pool = negative_pool.drop_duplicates(
        subset=["canonical_smiles"]
    )

    negative_pool["source_dataset"] = "ChEBI"

    NEGATIVE_POOL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    negative_pool.to_csv(
        NEGATIVE_POOL_PATH,
        index=False,
    )

    # -----------------------------
    # Deterministic balanced sample
    # -----------------------------
    target_size = len(positive)

    if len(negative_pool) < target_size:
        raise ValueError(
            "Not enough unique ChEBI negative candidates."
        )

    negative = negative_pool.sample(
        n=target_size,
        random_state=RANDOM_SEED,
    )

    modeling = pd.concat(
        [
            positive,
            negative,
        ],
        ignore_index=True,
    )

    modeling = modeling.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    MODELING_DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling.to_csv(
        MODELING_DATASET_PATH,
        index=False,
    )

    print("DrugCentral unique positives:", len(positive))
    print("ChEBI unique negative pool:", len(negative_pool))
    print("Balanced positives:", len(positive))
    print("Balanced negatives:", len(negative))
    print("Modeling dataset:", len(modeling))
    print("Random seed:", RANDOM_SEED)
    print("Negative-pool QC:", stats)
    print(
        "Negative pool:",
        NEGATIVE_POOL_PATH,
    )
    print(
        "Modeling dataset:",
        MODELING_DATASET_PATH,
    )


if __name__ == "__main__":
    main()