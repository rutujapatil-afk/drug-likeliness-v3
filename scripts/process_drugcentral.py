from pathlib import Path

import pandas as pd

from druglikeness.drugcentral import (
    load_drugcentral,
    process_drugcentral,
    summarize_drugcentral,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "structures.smiles.tsv"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "drugcentral_processed.csv"
)
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "drugcentral_summary.csv"
)


def main() -> None:
    dataframe = load_drugcentral(RAW_PATH)
    records = process_drugcentral(dataframe)

    processed = pd.DataFrame(
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

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = summarize_drugcentral(records)

    pd.DataFrame([summary]).to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(f"Processed dataset: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(summary)


if __name__ == "__main__":
    main()