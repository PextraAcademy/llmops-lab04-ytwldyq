from src.config import get_training_config


def test_training_config_contains_required_values():
    config = get_training_config()

    assert config is not None
    assert config["model_name"]
    assert config["learning_rate"] > 0
    assert config["num_epochs"] >= 1
    assert config["output_dir"]
