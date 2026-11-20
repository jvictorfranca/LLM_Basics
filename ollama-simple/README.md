
# Ollama Simple Agent

A local AI agent built with **Ollama, Python, FastAPI, and SQLite**, running with Docker Compose.

The goal is to allow a local LLM to execute Python functionalities through tools (function calling), such as retrieving users and creating new users directly in an SQLite database.

## Technologies

- **Python 3.12:** Main programming language.
- **FastAPI:** HTTP API framework.
- **Ollama:** Local LLM inference engine.
- **Qwen 2.5 7B:** Default language model.
- **SQLite:** Local database.
- **Docker:** Application containerization.
- **Docker Compose:** Service orchestration.

## Architecture

The project consists of two main services:

| Service | Description | Port |
|---|---|---|
| ollama | LLM inference server | 11434 |
| python-app | FastAPI application and Python agent | 8000 |

Python communicates with Ollama through the internal Docker Compose network.

The agent uses Python tools to perform operations on the SQLite database.

## Project Structure

```text
ollama-simple/
│
├── app/
│   ├── main.py
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   └── orchestrator.py
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   └── user_tools.py
│   │
│   └── database/
│       ├── __init__.py
│       └── connection.py
│
├── data/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
├── .env
└── .dockerignore
```

### Directories

- `app/agent`: Agent implementation and orchestration.
- `app/tools`: Python tools available to the agent.
- `app/database`: SQLite connection and configuration.
- `data/`: Persistent database storage.

## Configuration

Create a `.env` file in the project root.

Example:

```env
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_URL=http://ollama:11434
```

You can change the model according to your needs.

Example:

```env
OLLAMA_MODEL=llama3.1:8b
```

If no model is provided, Docker Compose will use the configured default model.

## Running the Project

### 1. Prerequisites

Install:

- Docker Desktop (Windows) or Docker Engine (Linux).
- Docker Compose V2.
- GNU Make (optional, for using shortcuts).

### 2. Start the containers

From the project root directory:

```bash
make up
```

Or directly using Docker Compose:

```bash
docker compose up -d
```

### 3. Build the application

To rebuild the Python image:

```bash
make build
```

Or:

```bash
docker compose up -d --build
```

### 4. Check running containers

```bash
make ps
```

Or:

```bash
docker compose ps
```

### 5. Stop the containers

```bash
make down
```

Or:

```bash
docker compose down
```

Docker volumes are preserved by the default command, keeping Ollama models and SQLite data.

## Available Makefile Commands

| Command | Description |
|---|---|
| `make up` | Start all services |
| `make down` | Stop and remove containers |
| `make restart` | Restart all services |
| `make build` | Rebuild and start services |
| `make ps` | List containers |
| `make logs` | Display logs from all services |
| `make logs-ollama` | Display Ollama logs |
| `make logs-python` | Display Python application logs |

Use `Ctrl + C` to stop following logs.

This only stops log streaming, not the containers themselves.

## Installing Make

GNU Make allows you to execute commands through simple shortcuts.

### Windows

There are several installation options.

**Option 1: Chocolatey**

Install Chocolatey following the official documentation:

https://chocolatey.org/install

Then open PowerShell as Administrator:

```powershell
choco install make
```

**Option 2: Winget**

In PowerShell:

```powershell
winget install GnuWin32.Make
```

After installation, close and reopen your terminal.

Verify the installation:

```powershell
make --version
```

**Important: GnuWin32 PATH configuration**

If the command is not recognized, check whether the installation's `bin` directory has been added to your Windows PATH.

The directory is usually:

```text
C:\Program Files (x86)\GnuWin32\bin
```

You can also check the installation with:

```powershell
where.exe make
```

If you are using Git Bash, restart it after updating the PATH.

### Linux

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install make
```

**Fedora:**

```bash
sudo dnf install make
```

**Arch Linux:**

```bash
sudo pacman -S make
```

Verify the installation:

```bash
make --version
```

## Testing the API

The application provides four main endpoints.

### 1. Traditional Chat

Sends a message directly to the language model without using agent tools.

```bash
curl "http://localhost:8000/chat?prompt=Hello,%20how%20are%20you?"
```

### 2. List Users

Retrieves all registered users from SQLite.

```bash
curl http://localhost:8000/users
```

### 3. Create User

Creates a user directly in the SQLite database.

```bash
curl -X POST http://localhost:8000/users \
  -H "Content-Type: application/json" \
  -d '{"name":"John","email":"john@example.com"}'
```

### 4. Execute the Agent

Sends an instruction to the agent, allowing it to decide which Python tool to execute.

Example: ask the agent to list registered users.

```bash
curl -G http://localhost:8000/agent \
  --data-urlencode "prompt=List all registered users"
```

Example: ask the agent to create a user.

```bash
curl -G http://localhost:8000/agent \
  --data-urlencode "prompt=Create a user named Maria with email maria@example.com"
```

The agent identifies the user's intent, calls the appropriate tool, and returns the execution result.

## GPU Support

Ollama can use an NVIDIA GPU to accelerate model inference.

GPU support depends on Docker Desktop configuration, NVIDIA drivers, and GPU runtime availability.

To check whether the model is using the GPU:

```bash
docker exec -it ollama ollama ps
```

The PROCESSOR column indicates where the model is running.

Example:

```text
NAME          SIZE     PROCESSOR
qwen2.5:7b    5.1 GB   100% GPU
```

## Ollama Models

To list installed models:

```bash
docker exec -it ollama ollama list
```

To download another model:

```bash
docker exec -it ollama ollama pull llama3.1:8b
```

Then update `OLLAMA_MODEL` in `.env` and recreate the services as needed.

## API Documentation

Once the containers are running, access:

**Swagger UI:**

http://localhost:8000/docs

**ReDoc:**

http://localhost:8000/redoc

## Data Persistence

The project uses Docker volumes to preserve data.

- `ollama_data`: Stores models downloaded by Ollama.
- `data/`: Stores the SQLite database, according to the application's volume configuration.

To also remove Docker volumes:

```bash
docker compose down -v
```

**Warning:** This command removes Compose-managed volumes and may delete downloaded models and persisted data.

## Troubleshooting

### Python cannot find modules

Check whether the directory structure is correct and whether your Dockerfile contains:

```dockerfile
COPY app/ .
```

Also make sure that the `agent`, `tools`, and `database` directories contain their respective `__init__.py` files.

### Ollama is not responding

Check the logs:

```bash
make logs-ollama
```

Check whether the container is running:

```bash
docker compose ps
```

### Python cannot connect to Ollama

Check the environment variables:

```env
OLLAMA_URL=http://ollama:11434
```

Inside the Docker network, the application must use the `ollama` service name instead of `localhost`.

### HTTP 500 errors

Check the application logs:

```bash
make logs-python
```

And:

```bash
make logs-ollama
```

### Model responses are slow

Check whether the GPU is being used:

```bash
docker exec -it ollama ollama ps
```

Larger models require more memory and processing power.

---

## Roadmap

- [ ] Add new tools to the agent.
- [ ] Implement conversation memory.
- [ ] Implement message history using SQLite.
- [ ] Add tools for additional database operations.
- [ ] Implement confirmation before destructive operations.
- [ ] Evolve the architecture into an MCP server.
- [ ] Add automated tests.
- [ ] Implement API authentication.

## Project Goal

This project is a learning environment for exploring local AI agents, function calling, Python tool integration, and self-hosted LLM infrastructure.

The goal is to gradually evolve from a traditional chatbot into an agent capable of performing real-world tasks through controlled tools.
