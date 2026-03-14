# MARS — Metacognitive AI Reasoning System

[![CI](https://github.com/Garrettc123/mars-production/actions/workflows/ci.yml/badge.svg)](https://github.com/Garrettc123/mars-production/actions/workflows/ci.yml)

MARS is a production-ready metacognitive AI reasoning engine built on FastAPI and
Claude (Anthropic). It exposes a clean REST API that allows clients to:

- **Reason** — ask the model to reason carefully about any query
- **Metacognize** — reflect on existing reasoning and surface improvements
- **Optimize** — run iterative self-improvement loops to progressively refine output

---

## Quickstart

### Prerequisites

- Python 3.11+
- An [Anthropic API key](https://console.anthropic.com/)

### 1. Clone & install

```bash
git clone https://github.com/Garrettc123/mars-production.git
cd mars-production
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and MARS_API_KEY
```

### 3. Run the server

```bash
uvicorn mars_api:app --reload
```

The API is now available at <http://localhost:8000>.  
Interactive docs: <http://localhost:8000/docs>

---

## Docker

```bash
# Build and run
docker compose up --build

# Or with plain Docker
docker build -t mars-api .
docker run -p 8000:8000 --env-file .env mars-api
```

---

## API Reference

All protected endpoints require the `X-Api-Key` header set to the value of
your `MARS_API_KEY` environment variable.

### `GET /health`

Liveness probe — returns `200 OK` with no authentication required.

```json
{ "status": "healthy", "service": "MARS" }
```

### `GET /api/status`

Runtime status and uptime information.

```json
{
  "service": "MARS",
  "version": "1.0.0",
  "uptime_seconds": 42.5,
  "timestamp": "2026-03-14T18:00:00+00:00",
  "model": "claude-3-5-sonnet-20241022"
}
```

### `POST /api/reason`

Send a query and receive reasoned output.

**Request**

```json
{
  "query": "What is the nature of consciousness?",
  "max_tokens": 4096
}
```

**Response**

```json
{
  "reasoning": "...",
  "model": "claude-3-5-sonnet-20241022",
  "success": true
}
```

### `POST /api/metacognize`

Reflect on existing reasoning and return an improved analysis.

**Request**

```json
{
  "reasoning": "<prior reasoning text>",
  "context": "optional extra context",
  "max_tokens": 4096
}
```

**Response**

```json
{
  "reflection": "...",
  "model": "claude-3-5-sonnet-20241022",
  "success": true
}
```

### `POST /api/optimize`

Run iterative self-improvement loops (1–5 iterations).

**Request**

```json
{
  "task": "Write a concise explanation of entropy",
  "iterations": 3,
  "max_tokens": 4096
}
```

**Response**

```json
{
  "final_output": "...",
  "iterations_completed": 3,
  "history": [
    { "iteration": 1, "output": "..." },
    { "iteration": 2, "output": "..." },
    { "iteration": 3, "output": "..." }
  ],
  "model": "claude-3-5-sonnet-20241022",
  "success": true
}
```

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                  Client                      │
└──────────────────────┬──────────────────────┘
                       │ HTTPS
┌──────────────────────▼──────────────────────┐
│             FastAPI (mars_api.py)            │
│  /health  /api/status  /api/reason           │
│  /api/metacognize  /api/optimize             │
└──────────────────────┬──────────────────────┘
                       │ Anthropic SDK
┌──────────────────────▼──────────────────────┐
│       Claude (claude-3-5-sonnet-20241022)    │
└─────────────────────────────────────────────┘
```

MARS implements a **metacognitive loop**:

1. **Reason** — initial response generation
2. **Metacognize** — self-reflection pass identifying weaknesses
3. **Optimize** — iterative refinement until quality threshold is reached

---

## Testing

```bash
pytest tests/ --cov=mars_api --cov-report=term-missing -v
```

The suite mocks all Anthropic API calls so no real credits are consumed.

---

## Deployment

### Railway

The included `railway.toml` configures automatic deployment from this
repository. Set `ANTHROPIC_API_KEY` and `MARS_API_KEY` in the Railway
project's environment variable settings.

### Render

Create a new **Web Service**, point it at this repository, and set:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn mars_api:app --host 0.0.0.0 --port $PORT`
- Environment variables: `ANTHROPIC_API_KEY`, `MARS_API_KEY`

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key for Claude access |
| `MARS_API_KEY` | ✅ | Shared secret for API authentication |
| `PORT` | ❌ | Server port (default: `8000`) |
