from __future__ import annotations

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray


def smiles_to_morgan(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Convert SMILES into a Morgan fingerprint."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    fingerprint = generator.GetFingerprint(molecule)

    array = np.zeros(
        n_bits,
        dtype=np.uint8,
    )

    ConvertToNumpyArray(
        fingerprint,
        array,
    )

    return array


def dataframe_to_morgan(
    dataframe,
    radius: int = 2,
    n_bits: int = 2048,
):
    """Convert a dataframe of molecules to Morgan fingerprints."""
    return np.vstack(
        [
            smiles_to_morgan(
                smiles,
                radius=radius,
                n_bits=n_bits,
            )
            for smiles in dataframe["canonical_smiles"]
        ]
    )