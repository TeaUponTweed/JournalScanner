.PHONY: polish
polish:
	black src
	isort src

.PHONY: freeze
freeze:
	pip-compile --resolver=backtracking

.PHONY: freeze
install:
	pip install -r dev-requirements.txt -r requirements.txt -e .
