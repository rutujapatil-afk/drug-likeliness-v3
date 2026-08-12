import json

from druglikeness.provenance import (
    DatasetMetadata,
    save_dataset_metadata,
)


def test_save_dataset_metadata(tmp_path):
    metadata = DatasetMetadata(
        source_id="drugcentral",
        source_url="https://drugcentral.org/download",
        acquired_date="2026-08-13",
        local_filename="drugcentral.smi",
        sha256="abc123",
        notes="Test dataset",
    )

    output = tmp_path / "metadata.json"

    save_dataset_metadata(
        metadata,
        output,
    )

    assert output.exists()

    saved = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert saved["source_id"] == "drugcentral"
    assert saved["sha256"] == "abc123"