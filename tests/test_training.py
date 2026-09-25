from src.train import create_trainer, save_model


def test_create_trainer_returns_trainer_like_object(monkeypatch):
    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(__import__("sys").modules, "transformers", type("T", (), {
        "Trainer": FakeTrainer,
        "TrainingArguments": lambda **kwargs: kwargs,
    })())

    trainer = create_trainer("model", "tokenizer", ["example"], {
        "output_dir": "artifacts/model",
        "num_epochs": 1,
        "learning_rate": 0.001,
    })

    assert trainer is not None
    assert trainer.kwargs["model"] == "model"
    assert trainer.kwargs["train_dataset"] == ["example"]


def test_save_model_writes_adapter_artifacts(tmp_path):
    class FakeModel:
        def save_pretrained(self, path):
            self.path = path

    class FakeTokenizer:
        def save_pretrained(self, path):
            self.path = path

    model = FakeModel()
    result = save_model(model, tmp_path)

    assert result is not None
    assert model.path == str(tmp_path)
