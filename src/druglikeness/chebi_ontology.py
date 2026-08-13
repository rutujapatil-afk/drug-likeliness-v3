from __future__ import annotations

import gzip
from pathlib import Path


DRUG_CHEBI_ID = "CHEBI:23888"
HAS_ROLE_RELATIONSHIP = "RO:0000087"


def load_chebi_drug_role_ids(
    path: str | Path,
) -> set[str]:
    """Return ChEBI IDs explicitly assigned the drug role."""
    path = Path(path)

    drug_role_ids: set[str] = set()
    current_id: str | None = None

    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line == "[Term]":
                current_id = None
                continue

            if not line or line.startswith("["):
                continue

            if line.startswith("id: "):
                current_id = line[4:].strip()
                continue

            if current_id is None:
                continue

            if line.startswith("relationship: "):
                parts = line.split()

                if (
                    len(parts) >= 3
                    and parts[1] == HAS_ROLE_RELATIONSHIP
                    and parts[2] == DRUG_CHEBI_ID
                ):
                    drug_role_ids.add(current_id)

    return drug_role_ids