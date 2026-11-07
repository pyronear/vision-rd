.PHONY: install lint format

install: ## Install dependencies
	uv sync

lint: ## Run ruff
	uv run ruff check .

format: ## Format with ruff
	uv run ruff format .
