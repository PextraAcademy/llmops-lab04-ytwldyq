"""Training pipeline for the LoRA adapter."""

def create_trainer(model, tokenizer, train_dataset, training_config):
    """Create a Hugging Face Trainer for the prepared dataset."""
    # TODO: STUDENT IMPLEMENT
    pass


def train_model():
    """Load data, prepare a LoRA model, train it, and save the adapter."""
    # TODO: STUDENT IMPLEMENT
    pass


def save_model(model, output_dir):
    """Save the trained adapter and tokenizer artifacts."""
    # TODO: STUDENT IMPLEMENT
    pass


if __name__ == "__main__":
    train_model()
