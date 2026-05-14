.PHONY: install lint format test help notebook audit-app audit-export audit-publish

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and set up notebook output stripping
	uv sync
	uv run nbstripout --install

lint: ## Run ruff linter on code and notebooks
	uv run ruff check .

format: ## Format code and notebooks with ruff
	uv run ruff format .

test: ## Run tests with pytest
	uv run pytest tests/ -v

notebook: ## Launch JupyterLab for notebooks
	uv run jupyter lab notebooks/

audit-app: ## Launch the bbox-editing audit app on http://localhost:8000
	uv run --group audit-app python scripts/run_audit_app.py

audit-export: ## Build YOLO-format patches under data/10_export/ from review.json
	uv run --group audit-app python scripts/export_audit_app.py
	@echo ""
	@echo "Next: make audit-publish    # detect dirty splits, push + commit per split"

audit-publish: ## Push + commit review/export per dirty split (interactive). MODEL=<name> to override.
	@bash scripts/audit_publish.sh

