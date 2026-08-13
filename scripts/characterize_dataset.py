from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


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
    / "dataset_characterization.csv"
)


def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "heavy_atoms": Lipinski.HeavyAtomCount(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": Lipinski.RingCount(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "logp": Crippen.MolLogP(mol),
    }


def main():
    dataframe = pd.read_csv(DATASET_PATH)

    descriptor_rows = [
        calculate_descriptors(smiles)
        for smiles in dataframe["canonical_smiles"]
    ]

    descriptors = pd.DataFrame(descriptor_rows)

    result = pd.concat(
        [
            dataframe[["canonical_smiles", "label"]],
            descriptors,
        ],
        axis=1,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("=== Dataset characterization ===")
    print("Rows:", len(result))

    print("\n=== Overall ===")
    print(
        result[
            [
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
        ].describe().round(3)
    )

    print("\n=== By class ===")

    summary = (
        result
        .groupby("label")[
            [
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
        ]
        .agg(["mean", "median", "std"])
        .round(3)
    )

    print(summary)

    print("\n=== Output ===")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()