import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

import anthropic
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import mars_router

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

VERSION = "1.1.0"

app = FastAPI(
    title="MARS API",
    description=(
        "Metacognitive AI Reasoning System — self-reflective reasoning engine "
        "with uncertainty-aware routing"
    ),
    version=VERSION,
)

_start_time: float = time.time()


def _get_client() -> anthropic.Anthropic:
    """Return a configured Anthropic client (lazy so tests can patch env vars)."""
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _require_api_key(x_api_key: Optional[str]) -> None:
    """Raise 401 if the provided key does not match MARS_API_KEY."""
    expected = os.getenv("MARS_API_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _upstream_error(exc: Exception) -> HTTPException:
    """Map an Anthropic API failure to a 502 without leaking request details."""
    return HTTPException(
        status_code=502, detail=f"Upstream model error: {exc.__class__.__name__}"
    )


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ReasonRequest(BaseModel):
    query: str
    # Caps the deep path only; the standard path uses MARS_STANDARD_MAX_TOKENS.
    max_tokens: Optional[int] = None
    # auto: cheap pass first, escalate to deep only when uncertain (default).
    # standard: cheap pass + score, never escalate. deep: always deep.
    route: Optional[Literal["auto", "standard", "deep"]] = None


class MetacognizeRequest(BaseModel):
    reasoning: str
    context: Optional[str] = None
    max_tokens: int = 4096


class OptimizeRequest(BaseModel):
    task: str
    iterations: int = 3
    max_tokens: int = 4096


class NWUPayload(BaseModel):
    nwu_version: str
    event_type: str
    source: str
    target: str
    auth_token: str
    payload: Dict[str, Any]
    timestamp: str


def _utility_call(client: Any, *, system: str, prompt: str, max_tokens: int):
    """Single call on the deep model for the metacognize / optimize endpoints."""
    cfg = mars_router.load_config()
    result = mars_router.call(
        client,
        model=cfg.deep_model,
        effort=cfg.utility_effort,
        max_tokens=max_tokens,
        system=system,
        query=prompt,
    )
    return result.text, cfg.deep_model


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health():
    """Simple liveness probe."""
    return {"status": "healthy", "service": "MARS"}


@app.get("/api/status", tags=["system"])
async def status():
    """Return runtime status, uptime, and routing configuration."""
    cfg = mars_router.load_config()
    uptime_seconds = round(time.time() - _start_time, 2)
    return {
        "service": "MARS",
        "version": VERSION,
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": cfg.standard_model,
        "models": {"standard": cfg.standard_model, "deep": cfg.deep_model},
        "routing": {
            "default_route": cfg.default_route,
            "threshold": cfg.threshold,
            "score_version": mars_router.SCORE_VERSION,
        },
    }


@app.post("/api/reason", tags=["reasoning"])
def reason(request: ReasonRequest, x_api_key: Optional[str] = Header(None)):
    """Primary reasoning endpoint with uncertainty-aware routing.

    Runs a cheap standard pass that reports its own confidence and escalates to the
    deep path only when uncertainty >= the threshold (or the pass is truncated or
    unscored). The `route` object in the response says which path answered and why.
    """
    _require_api_key(x_api_key)
    client = _get_client()
    try:
        result = mars_router.run(
            client, request.query, route=request.route, max_tokens=request.max_tokens
        )
    except anthropic.APIError as exc:
        raise _upstream_error(exc) from exc
    return {
        "reasoning": result.answer,
        "model": result.model,
        "success": True,
        "route": result.public(),
    }


@app.get("/api/routing/stats", tags=["reasoning"])
def routing_stats(x_api_key: Optional[str] = Header(None)):
    """In-memory routing counters since process start: how often the deep path runs."""
    _require_api_key(x_api_key)
    return mars_router.STATS.snapshot()


@app.post("/api/metacognize", tags=["reasoning"])
def metacognize(
    request: MetacognizeRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Metacognitive self-reflection endpoint."""
    _require_api_key(x_api_key)
    client = _get_client()
    context_block = f"\n\nAdditional context: {request.context}" if request.context else ""
    prompt = (
        f"Reflect on the following reasoning and identify any flaws, "
        f"blind spots, or improvements:{context_block}\n\n{request.reasoning}"
    )
    try:
        text, model = _utility_call(
            client,
            system=(
                "You are MARS (Metacognitive AI Reasoning System). "
                "Your role is to critically examine reasoning, identify weaknesses, "
                "and produce an improved, self-corrected analysis."
            ),
            prompt=prompt,
            max_tokens=request.max_tokens,
        )
    except anthropic.APIError as exc:
        raise _upstream_error(exc) from exc
    return {"reflection": text, "model": model, "success": True}


@app.post("/api/optimize", tags=["reasoning"])
def optimize(
    request: OptimizeRequest,
    x_api_key: Optional[str] = Header(None),
):
    """Iterative self-optimization endpoint (1–5 iterations)."""
    _require_api_key(x_api_key)
    iterations = max(1, min(request.iterations, 5))
    client = _get_client()

    current_output: str = ""
    history: list[dict] = []
    model = mars_router.load_config().deep_model

    for i in range(iterations):
        if i == 0:
            prompt = request.task
        else:
            prompt = (
                f"Previous attempt:\n{current_output}\n\n"
                f"Reflect on the above and produce a significantly improved version "
                f"for the original task: {request.task}"
            )
        try:
            current_output, model = _utility_call(
                client,
                system=(
                    "You are MARS (Metacognitive AI Reasoning System). "
                    "Each iteration you must improve upon the previous attempt."
                ),
                prompt=prompt,
                max_tokens=request.max_tokens,
            )
        except anthropic.APIError as exc:
            raise _upstream_error(exc) from exc
        history.append({"iteration": i + 1, "output": current_output})

    return {
        "final_output": current_output,
        "iterations_completed": iterations,
        "history": history,
        "model": model,
        "success": True,
    }


@app.post("/nwu-listener", tags=["nwu"])
async def nwu_listener(data: NWUPayload):
    """NWU Protocol ingress — preserve agent opportunity handshake from main."""
    expected_token = os.getenv("INTERNAL_AGENT_TOKEN")
    if not expected_token or data.auth_token != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized NWU payload")

    if data.event_type == "opportunity_detected":
        opportunity = data.payload
        return {
            "status": "acknowledged",
            "nwu_receipt": datetime.now(timezone.utc).isoformat(),
            "action": "queued_for_processing",
            "opportunity_id": opportunity.get("id") if isinstance(opportunity, dict) else None,
        }

    return {"status": "ignored", "reason": "unsupported_event_type"}
