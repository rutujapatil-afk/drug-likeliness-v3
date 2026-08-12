from __future__ import annotations

import hashlib
from pathlib import Path

import requests


CHUNK_SIZE = 1024 * 1024


def calculate_sha256(path: str | Path) -> str:
    """
    Calculate the SHA-256 checksum of a file.

    Parameters
    ----------
    path:
        Path to the file.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 checksum.
    """
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def download_file(
    url: str,
    destination: str | Path,
    timeout: int = 60,
) -> Path:
    """
    Download a file from a URL.

    Parameters
    ----------
    url:
        Source URL.
    destination:
        Local destination path.
    timeout:
        HTTP timeout in seconds.

    Returns
    -------
    pathlib.Path
        Path to the downloaded file.
    """
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    response = requests.get(
        url,
        stream=True,
        timeout=timeout,
    )

    response.raise_for_status()

    with destination.open("wb") as file:
        for chunk in response.iter_content(
            chunk_size=CHUNK_SIZE
        ):
            if chunk:
                file.write(chunk)

    return destination