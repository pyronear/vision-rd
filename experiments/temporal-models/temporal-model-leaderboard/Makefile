.PHONY: install lint format test help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

lint: ## Run ruff linter on code
	uv run ruff check src/ tests/ scripts/

format: ## Format code with ruff
	uv run ruff format src/ tests/ scripts/

test: ## Run tests with pytest
	uv run pytest tests/ -v
