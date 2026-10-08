PYTHON ?= python3
export PYTHONPATH := src
.PHONY: test demo lint
test:
	$(PYTHON) -m unittest discover -s tests -v
demo:
	$(PYTHON) scripts/run_episode.py 19 3
lint:
	$(PYTHON) -m ruff check src tests scripts
