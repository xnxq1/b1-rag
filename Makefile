.PHONY: help env install build up down restart logs logs-api logs-worker ps shell-api clean infra run-api run-worker test test-unit test-integration lint format sample examples
.DEFAULT_GOAL := help

help: ## Показать помощь
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

env: ## Создать .env из .env.example (если его нет)
	@test -f .env && echo ".env уже есть" || (cp .env.example .env && echo ".env создан — укажите OPENROUTER_API_KEY")

install: ## Установить зависимости локально
	uv sync

build: ## Собрать образы
	docker compose build

up: ## Поднять все сервисы в фоне
	docker compose up -d --build

down: ## Остановить сервисы
	docker compose down

restart: ## Перезапустить api и worker
	docker compose restart api worker

logs: ## Логи всех сервисов
	docker compose logs -f

logs-api: ## Логи api
	docker compose logs -f api

logs-worker: ## Логи worker
	docker compose logs -f worker

ps: ## Статус контейнеров
	docker compose ps

shell-api: ## Shell в api
	docker compose exec api /bin/bash

clean: ## Удалить контейнеры и volumes (Qdrant, Redis, загруженные файлы)
	docker compose down -v

infra: ## Поднять только Qdrant и Redis (для локального запуска)
	docker compose up -d qdrant redis

run-api: ## Запустить API локально
	uv run python manage.py start-api

run-worker: ## Запустить воркер локально
	uv run python manage.py start-worker

test: ## Все тесты (интеграционным нужен Docker)
	uv run pytest

test-unit: ## Юнит-тесты
	uv run pytest tests/unit

test-integration: ## Интеграционные тесты (Qdrant, Redis через testcontainers)
	uv run pytest tests/integration

lint: ## ruff check + проверка форматирования + mypy
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy rag manage.py

format: ## ruff fix + format
	uv run ruff check . --fix
	uv run ruff format .

sample: ## Сгенерировать демо-договор examples/contract.pdf
	uv run python examples/make_sample.py

examples: ## Прогнать примеры из README против запущенного сервиса
	./examples/run_examples.sh
