.PHONY: clean
.DEFAULT_GOAL := install

PATHS_TO_FORMAT=rf_statistics tests
PATHS_TO_TESTS=tests

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .pytest_cache
	find . -name '*.egg-info' -exec rm -fr {} +

install: clean
	poetry install

format:
	black ${PATHS_TO_FORMAT}

format-check:
	black --check ${PATHS_TO_FORMAT}

lint: format-check
	yamllint .

test-quick:
	pytest -vvv -s -m "not slow" ${PATHS_TO_TESTS}

test-all:
	pytest -vvv -s ${PATHS_TO_TESTS}

test-report:
	pytest --cov=src --cov-report=term-missing -vvvs --junitxml=pytest.xml ${PATHS_TO_TESTS}

test: test-quick

make bc: format lint
