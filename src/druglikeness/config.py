from pathlib import Path

import yaml


def load_yaml_config(path: str | Path) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML configuration.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        configuration = yaml.safe_load(file)

    if not isinstance(configuration, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return configuration