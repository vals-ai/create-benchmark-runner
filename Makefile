.PHONY: install lock sync test lint typecheck help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install deps
	uv sync

lock: ## Re-lock pyproject deps
	uv lock

sync: ## uv sync only
	uv sync

test: ## Run tests
	uv run pytest -q

lint: ## Run ruff
	uv run ruff check .

typecheck: ## Run basedpyright
	uv run basedpyright
