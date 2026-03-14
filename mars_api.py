import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import anthropic

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MARS API",
    description="Metacognitive AI Reasoning System — self-reflective reasoning engine",
    version="1.0.0",
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


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

MODEL = "claude-3-5-sonnet-20241022"


class ReasonRequest(BaseModel):
    query: str
    max_tokens: int = 4096


class MetacognizeRequest(BaseModel):
    reasoning: str
    context: Optional[str] = None
    max_tokens: int = 4096


class OptimizeRequest(BaseModel):
    task: str
    iterations: int = 3
    max_tokens: int = 4096


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health():
    """Simple liveness probe."""
    return {"status": "healthy", "service": "MARS"}


@app.get("/api/status", tags=["system"])
async def status():
    """Return runtime status and uptime information."""
    uptime_seconds = round(time.time() - _start_time, 2)
    return {
        "service": "MARS",
        "version": "1.0.0",
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
    }


@app.post("/api/reason", tags=["reasoning"])
async def reason(request: ReasonRequest, x_api_key: Optional[str] = Header(None)):
    """
    Primary reasoning endpoint.

    Sends the query to the underlying language model and returns its response
    as structured reasoning output.
    """
    _require_api_key(x_api_key)
    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=request.max_tokens,
        system=(
            "You are MARS (Metacognitive AI Reasoning System). "
            "Reason carefully and thoroughly about the user's query."
        ),
        messages=[{"role": "user", "content": request.query}],
    )
    return {
        "reasoning": response.content[0].text,
        "model": MODEL,
        "success": True,
    }


@app.post("/api/metacognize", tags=["reasoning"])
async def metacognize(
    request: MetacognizeRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    Metacognitive self-reflection endpoint.

    Given a piece of prior reasoning, MARS reflects on its own thought process,
    identifies gaps or errors, and returns an improved analysis.
    """
    _require_api_key(x_api_key)
    client = _get_client()
    context_block = f"\n\nAdditional context: {request.context}" if request.context else ""
    prompt = (
        f"Reflect on the following reasoning and identify any flaws, "
        f"blind spots, or improvements:{context_block}\n\n{request.reasoning}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=request.max_tokens,
        system=(
            "You are MARS (Metacognitive AI Reasoning System). "
            "Your role is to critically examine reasoning, identify weaknesses, "
            "and produce an improved, self-corrected analysis."
        ),
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "reflection": response.content[0].text,
        "model": MODEL,
        "success": True,
    }


@app.post("/api/optimize", tags=["reasoning"])
async def optimize(
    request: OptimizeRequest,
    x_api_key: Optional[str] = Header(None),
):
    """
    Iterative self-optimization endpoint.

    Runs multiple self-reflection loops to progressively refine a response to
    the given task, implementing the MARS metacognitive improvement cycle.
    """
    _require_api_key(x_api_key)
    iterations = max(1, min(request.iterations, 5))  # clamp to [1, 5]
    client = _get_client()

    current_output: str = ""
    history: list[dict] = []

    for i in range(iterations):
        if i == 0:
            prompt = request.task
        else:
            prompt = (
                f"Previous attempt:\n{current_output}\n\n"
                f"Reflect on the above and produce a significantly improved version "
                f"for the original task: {request.task}"
            )

        response = client.messages.create(
            model=MODEL,
            max_tokens=request.max_tokens,
            system=(
                "You are MARS (Metacognitive AI Reasoning System). "
                "Each iteration you must improve upon the previous attempt."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        current_output = response.content[0].text
        history.append({"iteration": i + 1, "output": current_output})

    return {
        "final_output": current_output,
        "iterations_completed": iterations,
        "history": history,
        "model": MODEL,
        "success": True,
    }
