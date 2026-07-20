format:
	ruff format src
	ruff check --fix src

lint:
	ruff check src
	ty check src

test:
	pytest

.PHONY: format lint test
