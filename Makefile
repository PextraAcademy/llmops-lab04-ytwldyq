PYTHON ?= python3

.PHONY: install lint test check train evaluate

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -v

check: lint test

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate
