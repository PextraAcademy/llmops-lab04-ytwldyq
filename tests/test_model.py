from src.model import (
    create_lora_config,
    load_base_model,
    load_tokenizer,
    prepare_model_for_training,
)


def test_lora_configuration_uses_parameter_efficient_settings():
    config = create_lora_config()

    assert config is not None
    assert config.r > 0
    assert config.lora_alpha >= config.r
    assert config.task_type is not None


def test_model_helpers_delegate_to_hugging_face(monkeypatch):
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            return {"tokenizer": name}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name):
            return {"model": name}

    monkeypatch.setitem(__import__("sys").modules, "transformers", type("T", (), {
        "AutoTokenizer": FakeAutoTokenizer,
        "AutoModelForCausalLM": FakeAutoModel,
    })())

    assert load_tokenizer("demo")["tokenizer"] == "demo"
    assert load_base_model("demo")["model"] == "demo"


def test_prepare_model_for_training_returns_peft_ready_model():
    class FakeModel:
        is_prepared = False

    result = prepare_model_for_training(FakeModel())

    assert result is not None
    assert getattr(result, "is_prepared", False)
