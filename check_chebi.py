import gzip

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

path = "data/raw/chebi/chebi.sdf.gz"

total = 0
unreadable = 0
with_smiles = 0

with gzip.open(path, "rb") as file:
    supplier = Chem.ForwardSDMolSupplier(
        file,
        sanitize=False,
        removeHs=False,
        strictParsing=False,
    )

    for molecule in supplier:
        total += 1

        if molecule is None:
            unreadable += 1
            continue

        if molecule.HasProp("SMILES"):
            with_smiles += 1

print("Total SDF records:", total)
print("Unreadable records:", unreadable)
print("Records with SMILES:", with_smiles)