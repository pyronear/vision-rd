.PHONY: install lint format test app help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

test: ## Run tests
	uv run pytest tests/ -v

app: ## Launch the Streamlit viewer
	uv run streamlit run src/temporal_model_explorer/app.py
