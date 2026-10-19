.PHONY: help install venv lint format typecheck test up down logs build migrate

help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies using uv"
	@echo "  make lint       Run ruff check"
	@echo "  make format     Run ruff format"
	@echo "  make typecheck  Run mypy type checker"
	@echo "  make test       Run pytest suite"
	@echo "  make up         Start docker compose services"
	@echo "  make down       Stop docker compose services"
	@echo "  make logs       View docker compose logs"
	@echo "  make build      Build docker images"

venv:
	uv venv .venv

install: venv
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy app

test:
	uv run pytest

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

build:
	docker compose build
