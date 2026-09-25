from src.evaluate import evaluate_model, predict


def test_predict_returns_a_label():
    class FakeTokenizer:
        def __call__(self, text, return_tensors):
            return {"input_ids": [[1]]}

        def decode(self, tokens, skip_special_tokens):
            return "database"

    class FakeModel:
        def generate(self, **inputs):
            return [[1, 2]]

    result = predict(FakeModel(), FakeTokenizer(), "Database connection failed")

    assert result == "database"


def test_evaluation_result_contains_counts_and_accuracy():
    class FakeModel:
        pass

    result = evaluate_model(
        FakeModel(),
        object(),
        [
            {"text": "one", "label": "database"},
            {"text": "two", "label": "networking"},
        ],
    )

    assert result["total_examples"] == 2
    assert 0 <= result["correct_predictions"] <= 2
    assert 0 <= result["accuracy"] <= 1
