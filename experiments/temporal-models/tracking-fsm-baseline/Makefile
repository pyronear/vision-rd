.PHONY: install lint format test help fiftyone-build fiftyone

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and set up notebook output stripping
	uv sync --group explore
	uv run nbstripout --install

lint: ## Run ruff linter on code and notebooks
	uv run ruff check .

format: ## Format code and notebooks with ruff
	uv run ruff format .

test: ## Run tests with pytest
	uv run pytest tests/ -v

fiftyone-build: ## Build FiftyOne datasets for train and val splits
	uv run --group explore python scripts/build_fiftyone_dataset.py \
		--split val \
		--track-dir data/07_model_output/val \
		--infer-dir data/03_primary/val \
		--data-dir data/01_raw/datasets/val
	uv run --group explore python scripts/build_fiftyone_dataset.py \
		--split train \
		--track-dir data/07_model_output/train \
		--infer-dir data/03_primary/train \
		--data-dir data/01_raw/datasets/train

fiftyone: fiftyone-build ## Build all datasets and launch FiftyOne app
	uv run --group explore python scripts/launch_fiftyone.py
