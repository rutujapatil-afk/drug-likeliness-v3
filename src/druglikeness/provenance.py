from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetMetadata:
    """Provenance metadata for an acquired dataset."""

    source_id: str
    source_url: str
    acquired_date: str
    local_filename: str
    sha256: str
    notes: str = ""


def save_dataset_metadata(
    metadata: DatasetMetadata,
    path: str | Path,
) -> Path:
    """
    Save dataset provenance metadata as JSON.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(metadata),
            file,
            indent=2,
        )

    return path