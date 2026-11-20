
.PHONY: up down restart logs-ollama logs-python logs ps build

# Start all services
up:
	docker compose up -d

# Stop and remove containers
down:
	docker compose down

# Restart all services
restart:
	docker compose down
	docker compose up -d

# Show Ollama logs
logs-ollama:
	docker compose logs -f ollama

# Show Python application logs
logs-python:
	docker compose logs -f python-app

# Show all services logs
logs:
	docker compose logs -f

# Show running containers
ps:
	docker compose ps

# Rebuild and start services
build:
	docker compose up -d --build
