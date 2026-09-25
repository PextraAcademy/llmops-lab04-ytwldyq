from src.data import load_dataset, prepare_example, tokenize_dataset


def test_load_dataset():
    dataset = load_dataset("data/train.json")

    assert len(dataset) > 0
    assert {"text", "label"} <= set(dataset[0])


def test_prepare_example():
    result = prepare_example(
        {"text": "Database connection failed.", "label": "database"}
    )

    assert result["text"] == "Database connection failed."
    assert result["label"] == "database"


def test_tokenize_dataset_uses_tokenizer():
    class FakeTokenizer:
        def __call__(self, text, truncation, max_length, padding):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    result = tokenize_dataset(
        [{"text": "A problem", "label": "networking"}], FakeTokenizer()
    )

    assert len(result) == 1
    assert "input_ids" in result[0]
    assert "labels" in result[0]
