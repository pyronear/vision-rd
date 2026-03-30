CRON_ENTRY = 0 8-20 * * * cd $(CURDIR) && $(shell which uv) run python scripts/capture_once.py >> data/01_raw/capture.log 2>&1

.PHONY: install lint format test capture cron-install cron-remove cron-show help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and set up notebook output stripping
	uv sync
	uv run nbstripout --install

lint: ## Run ruff linter on code and notebooks
	uv run ruff check .
	@find notebooks/ -name '*.ipynb' 2>/dev/null | grep -q . && uv run nbqa ruff check notebooks/ || true

format: ## Format code and notebooks with ruff
	uv run ruff format .
	@find notebooks/ -name '*.ipynb' 2>/dev/null | grep -q . && uv run nbqa ruff format notebooks/ || true

test: ## Run tests with pytest
	uv run pytest tests/ -v

capture: ## Run a single capture now (all cameras, all poses)
	uv run python scripts/capture_once.py

cron-install: ## Install hourly cron job (8am–8pm, logs to data/01_raw/capture.log)
	(crontab -l 2>/dev/null | grep -v "image-capture"; echo "$(CRON_ENTRY)") | crontab -
	@echo "Installed. Run 'make cron-show' to verify."

cron-remove: ## Remove the image-capture cron job
	crontab -l 2>/dev/null | grep -v "image-capture" | crontab -
	@echo "Removed."

cron-show: ## Show current crontab
	@crontab -l 2>/dev/null || echo "(empty crontab)"
