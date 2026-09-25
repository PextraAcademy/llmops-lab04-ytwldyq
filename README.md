# LLMOps Assignment 2: ModelOps

Assignment 2 builds on PromptOps from Assignment 1 by adapting a lightweight Hugging Face language model with parameter-efficient LoRA fine-tuning.

## Objective

You will build the workflow:

```text
Dataset -> Prepare -> Tokenize -> Fine-Tune -> Save Model -> Evaluate -> Compare
```

The task is software issue classification using a small labeled dataset. The instructor-selected starter model is configured as `sshleifer/tiny-gpt2` in `src/config.py`. Do not replace it unless instructed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
```

## Test-Driven Workflow

Start by running the tests. The starter repository intentionally contains real failing tests because many functions contain `# TODO: STUDENT IMPLEMENT` placeholders.

```bash
make test
```

Then:

1. Read each failing test.
2. Find the associated TODO in `src/`.
3. Ask GitHub Copilot to explain or implement the missing behavior.
4. Implement and review the code.
5. Run `make test` again.
6. Run linting and the complete check.

Unit tests use small fakes and mocks where possible. They should not download a model or perform a training run.

## Required Commands

```bash
make install
make test
make lint
make check
make train
make evaluate
```

Run `make check` before submitting. All tests and Ruff checks must pass.

## Assignment Tasks

Complete the TODOs in:

- `src/config.py`: return training settings.
- `src/data.py`: load, validate, and tokenize examples.
- `src/model.py`: load the model and tokenizer, configure PEFT/LoRA, and prepare the model.
- `src/train.py`: create a Trainer, run training, and save adapter artifacts.
- `src/evaluate.py`: load the adapter, predict labels, and calculate evaluation results.
- `tests/`: complete or improve the tests marked with TODOs as part of your engineering work.

Use LoRA/PEFT rather than full-model fine-tuning. Keep the dataset and training settings small enough for modest hardware.

## Evaluation Report

After training and evaluation, record your results here:

```text
Base Model
Accuracy: XX%

Fine-Tuned Model
Accuracy: XX%
```

Answer briefly:

1. Did fine-tuning improve the results?
2. Which examples were difficult?
3. What effect did the training data have?
4. What would you change in the dataset?
5. What are the limitations of this experiment?

The goal is to understand the ModelOps workflow, not to achieve a particular accuracy target.

## Copilot Policy

GitHub Copilot may help explain Hugging Face, datasets, PEFT/LoRA, Trainer, test failures, and Ruff errors. You remain responsible for reviewing, understanding, and validating the generated code.
