"""Evaluation helpers for base and fine-tuned models."""

from . import config
from .data import load_dataset


def load_finetuned_model(model_name, adapter_path):
    """Load a base model with a saved PEFT adapter."""
    # TODO: STUDENT IMPLEMENT
    pass


def predict(model, tokenizer, text):
    """Generate a label prediction for one issue."""
    # TODO: STUDENT IMPLEMENT
    pass


def evaluate_model(model, tokenizer, test_data):
    """Calculate total examples, correct predictions, and accuracy."""
    # TODO: STUDENT IMPLEMENT
    pass


if __name__ == "__main__":
    test_data = load_dataset(config.TEST_PATH)
    model, tokenizer = load_finetuned_model(config.MODEL_NAME, config.OUTPUT_DIR)
    print(evaluate_model(model, tokenizer, test_data))
