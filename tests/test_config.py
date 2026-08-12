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

    assert configuration["classes"]["positive"]["value"] == 1
    assert configuration["classes"]["positive"]["name"] == "drug_reference"

    assert configuration["classes"]["negative"]["value"] == 0
    assert configuration["classes"]["negative"]["name"] == "non_drug_reference"

    assert (
        configuration["molecular_representation"]["toolkit"]
        == "rdkit"
    )