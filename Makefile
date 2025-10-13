.PHONY: setup test smoke format lint run precommit report bench mwr ablate sweep exp-report docker paper clean

VENV ?= .venv
PYTHON ?= python3
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python
BLACK := $(VENV)/bin/black
ISORT := $(VENV)/bin/isort
RUFF := $(VENV)/bin/ruff
FLAKE8 := $(VENV)/bin/flake8
PYTEST := $(VENV)/bin/pytest
PRECOMMIT := $(VENV)/bin/pre-commit

SETUP_PATHS := cli feedflipnets scripts tests
SMOKE_PRESETS := mnist_mlp_dfa ucr_gunpoint_mlp_dfa california_housing_mlp_dfa 20newsgroups_bow_mlp_dfa

setup:
	@[ -d $(VENV) ] || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-lock.txt
	$(PRECOMMIT) install

format:
	$(ISORT) $(SETUP_PATHS)
	$(BLACK) $(SETUP_PATHS)

lint:
	$(RUFF) check $(SETUP_PATHS)
	$(FLAKE8) $(SETUP_PATHS)
	$(BLACK) --check $(SETUP_PATHS)

precommit:
	$(PRECOMMIT) run --all-files

test:
	FEEDFLIP_DATA_OFFLINE=1 PYTHONPATH=. $(PYTEST) -q

smoke:
	@for preset in $(SMOKE_PRESETS); do \
		echo "==> $$preset"; \
		FEEDFLIP_DATA_OFFLINE=1 PYTHONPATH=. $(PYTHON_BIN) -m cli.main --preset $$preset --offline; \
	done

report:
	@echo "Aggregating results to data/report/"
	FEEDFLIP_DATA_OFFLINE=1 PYTHONPATH=. $(PYTHON_BIN) scripts/aggregate_modality_results.py

bench:
	@echo "Running comprehensive benchmark sweep (this may take a while)"
	PYTHONPATH=. $(PYTHON_BIN) scripts/run_benchmark.py $(ARGS)
	@echo "Compiling benchmark report"
	PYTHONPATH=. $(PYTHON_BIN) scripts/compile_benchmark_report.py

mwr:
	PYTHONPATH=. $(PYTHON_BIN) -m exp.runner run --subset MWR

ablate:
	PYTHONPATH=. $(PYTHON_BIN) -m exp.runner run --subset ABLATE

sweep:
	PYTHONPATH=. $(PYTHON_BIN) -m exp.runner run --subset SWEEP

exp-report:
	PYTHONPATH=. $(PYTHON_BIN) -m exp.runner report --out runs/report.md

docker:
	docker build -t feedflipnets:exp -f exp/Dockerfile .

paper:
	@echo "Building LaTeX paper (requires pdflatex)"
	cd docs/paper && pdflatex -interaction=nonstopmode main.tex || true
	cd docs/paper && bibtex main || true
	cd docs/paper && pdflatex -interaction=nonstopmode main.tex || true
	cd docs/paper && pdflatex -interaction=nonstopmode main.tex || true
	@echo "Done. Output at docs/paper/main.pdf (if LaTeX is installed)."

paper-bundle:
	@echo "Creating paper bundle (tar.gz)"
	@mkdir -p dist
	@tar -czf dist/paper_bundle.tar.gz \
		docs/paper/main.tex \
		docs/paper/references.bib \
		data/report/best_configs_table.md \
		data/report/best_configs_table.tex \
		data/report/best_configs.csv \
		data/report/benchmark_summary.md \
		data/report/benchmark_summary.csv \
		data/report/plots || true
	@echo "Bundle at dist/paper_bundle.tar.gz"

run:
	@if [ -z "$(PRESET)" ]; then \
		echo "Usage: make run PRESET=<preset> [EXTRA_ARGS='--feedback dfa']"; \
		exit 1; \
	fi
	FEEDFLIP_DATA_OFFLINE=1 PYTHONPATH=. $(PYTHON_BIN) -m cli.main --preset $(PRESET) $(EXTRA_ARGS)

clean:
	rm -rf $(VENV)
