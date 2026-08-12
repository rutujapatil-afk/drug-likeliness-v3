from hashlib import sha256
from pathlib import Path

from druglikeness.acquisition import calculate_sha256


def test_calculate_sha256(tmp_path: Path):
    test_file = tmp_path / "test.txt"

    content = b"drug-likeliness-v3"

    test_file.write_bytes(content)

    expected = sha256(content).hexdigest()

    assert calculate_sha256(test_file) == expected