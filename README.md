# MARS — Metacognitive AI Reasoning System

Production FastAPI service for self-reflective reasoning + NWU Protocol ingress.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | none | Liveness |
| GET | `/api/status` | none | Uptime + model |
| POST | `/api/reason` | `X-Api-Key` | Primary reasoning |
| POST | `/api/metacognize` | `X-Api-Key` | Self-reflection |
| POST | `/api/optimize` | `X-Api-Key` | Iterative improve (1–5) |
| POST | `/nwu-listener` | `auth_token` in body | NWU opportunity handshake |

## Quickstart

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY, MARS_API_KEY, INTERNAL_AGENT_TOKEN
pip install -r requirements.txt
uvicorn mars_api:app --reload
```

## Docker / Railway

```bash
docker compose up --build
```

`railway.toml` points healthcheck at `/health`.

## Tests

```bash
pytest tests/ --cov=mars_api --cov-fail-under=80 -v
```

All Anthropic calls are mocked. No live credits required.
