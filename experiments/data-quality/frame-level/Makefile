.PHONY: install lint format test help notebook review-app review-export

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

review-app: ## Launch the bbox-editing review app on http://localhost:8000
	uv run --group review-app python scripts/run_review_app.py

review-export: ## Build YOLO-format patches under data/10_export/ from review.json
	uv run --group review-app python scripts/export_review_app.py
	@echo ""
	@echo "Next steps to share the export:"
	@echo "  uv run dvc add data/10_export && uv run dvc push"
	@echo "  git add data/10_export.dvc data/.gitignore && git commit -m 'export: refresh corrections'"

