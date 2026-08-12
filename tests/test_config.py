from pathlib import Path

from druglikeness.config import load_yaml_config


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "dataset.yaml"
)


def test_load_dataset_config():
    configuration = load_yaml_config(CONFIG_PATH)

    assert configuration["project"]["name"] == "drug-likeliness-v3"
    assert configuration["project"]["task"] == "binary_classification"

    assert configuration["target"]["name"] == "drug_likeness"
    assert configuration["target"]["type"] == "binary"

    assert configuration["classes"]["positive"] == 1
    assert configuration["classes"]["negative"] == 0

    assert (
        configuration["molecular_representation"]["toolkit"]
        == "rdkit"
    )