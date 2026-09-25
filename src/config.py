"""Configuration for the lightweight ModelOps experiment."""

MODEL_NAME = "sshleifer/tiny-gpt2"
TRAIN_PATH = "data/train.json"
TEST_PATH = "data/test.json"
OUTPUT_DIR = "artifacts/model"
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_LENGTH = 128


def get_training_config():
    """Return the training settings used by the assignment."""
    # TODO: STUDENT IMPLEMENT
    pass
