.PHONY: install pull push help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies from uv.lock
	uv sync

pull: ## Pull PDF data from S3 via DVC
	uv run dvc pull

push: ## Push PDF data to S3 via DVC
	uv run dvc push
