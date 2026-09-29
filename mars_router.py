"""Uncertainty-aware routing for MARS.

Cheap standard pass first. Escalate to the deep path when the pass is
unscored, truncated, or uncertainty >= threshold.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

SCORE_VERSION = "conf-re-v1.1"

_CONF_RE = re.compile(
    r"(?im)^[\s*_`>-]*confidence[\s*_`]*[:=][\s*_`]*([0-9]*\.?[0-9]+)\s*(%?)[\s*_`.]*$"
)

RouteName = Literal["auto", "standard", "deep"]


@dataclass(frozen=True)
class RouterConfig:
    standard_model: str
    deep_model: str
    threshold: float
    default_route: RouteName
    standard_max_tokens: int
    deep_max_tokens: int
    utility_effort: str
    standard_effort: str
    deep_effort: str


@dataclass
class CallResult:
    text: str
    model: str
    stop_reason: Optional[str] = None
    truncated: bool = False


@dataclass
class RouteResult:
    answer: str
    model: str
    path: str
    reason: str
    confidence: Optional[float] = None
    uncertainty: Optional[float] = None
    score_version: str = SCORE_VERSION
    escalated: bool = False

    def public(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "model": self.model,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "score_version": self.score_version,
            "escalated": self.escalated,
        }


class _Stats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.standard = 0
        self.deep = 0
        self.escalated = 0
        self.unscored = 0
        self.truncated = 0

    def record(self, *, path: str, escalated: bool, unscored: bool, truncated: bool) -> None:
        with self._lock:
            self.requests += 1
            if path == "deep":
                self.deep += 1
            else:
                self.standard += 1
            if escalated:
                self.escalated += 1
            if unscored:
                self.unscored += 1
            if truncated:
                self.truncated += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests": self.requests,
                "standard": self.standard,
                "deep": self.deep,
                "escalated": self.escalated,
                "unscored": self.unscored,
                "truncated": self.truncated,
                "deep_rate": (self.deep / self.requests) if self.requests else 0.0,
                "score_version": SCORE_VERSION,
            }


STATS = _Stats()


def load_config() -> RouterConfig:
    default_route = os.getenv("MARS_DEFAULT_ROUTE", "auto").strip().lower()
    if default_route not in {"auto", "standard", "deep"}:
        default_route = "auto"
    return RouterConfig(
        standard_model=os.getenv("MARS_STANDARD_MODEL", "claude-3-5-haiku-20241022"),
        deep_model=os.getenv("MARS_DEEP_MODEL", "claude-3-5-sonnet-20241022"),
        threshold=float(os.getenv("MARS_THRESHOLD", "0.50")),
        default_route=default_route,  # type: ignore[arg-type]
        standard_max_tokens=int(os.getenv("MARS_STANDARD_MAX_TOKENS", "1200")),
        deep_max_tokens=int(os.getenv("MARS_DEEP_MAX_TOKENS", "4096")),
        utility_effort=os.getenv("MARS_UTILITY_EFFORT", "high"),
        standard_effort=os.getenv("MARS_STANDARD_EFFORT", "low"),
        deep_effort=os.getenv("MARS_DEEP_EFFORT", "high"),
    )


def parse_confidence(text: str) -> Optional[float]:
    """Extract a 0-1 confidence from a trailing `confidence: N` line."""
    if not text:
        return None
    for raw in reversed(text.strip().splitlines()):
        line = raw.strip().strip("*`>_ ").rstrip(".")
        match = _CONF_RE.match(raw) or _CONF_RE.match(line)
        if not match:
            continue
        value = float(match.group(1))
        if match.group(2) == "%" or value > 1.0:
            value = value / 100.0
        if 0.0 <= value <= 1.0:
            return value
    return None


def _strip_confidence_line(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for line in lines:
        if _CONF_RE.match(line.strip()) or _CONF_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _is_truncated(stop_reason: Optional[str]) -> bool:
    return stop_reason in {"max_tokens", "length"}


def call(
    client: Any,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    system: str,
    query: str,
) -> CallResult:
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": query}],
    }
    try:
        response = client.messages.create(**kwargs, extra_headers={"x-mars-effort": effort})
    except TypeError:
        response = client.messages.create(**kwargs)

    text = ""
    if getattr(response, "content", None):
        block = response.content[0]
        text = getattr(block, "text", "") or ""
    stop_reason = getattr(response, "stop_reason", None)
    return CallResult(
        text=text,
        model=model,
        stop_reason=stop_reason,
        truncated=_is_truncated(stop_reason),
    )


_STANDARD_SYSTEM = (
    "You are MARS (Metacognitive AI Reasoning System). "
    "Reason carefully. End your reply with a single line exactly in this form:\n"
    "confidence: <0-1 or 0-100%>"
)

_DEEP_SYSTEM = (
    "You are MARS (Metacognitive AI Reasoning System). "
    "Reason thoroughly and produce the best possible answer. "
    "End your reply with a single line exactly in this form:\n"
    "confidence: <0-1 or 0-100%>"
)


def run(
    client: Any,
    query: str,
    route: Optional[RouteName] = None,
    max_tokens: Optional[int] = None,
) -> RouteResult:
    cfg = load_config()
    chosen: RouteName = route or cfg.default_route

    if chosen == "deep":
        deep = call(
            client,
            model=cfg.deep_model,
            effort=cfg.deep_effort,
            max_tokens=max_tokens or cfg.deep_max_tokens,
            system=_DEEP_SYSTEM,
            query=query,
        )
        conf = parse_confidence(deep.text)
        result = RouteResult(
            answer=_strip_confidence_line(deep.text),
            model=cfg.deep_model,
            path="deep",
            reason="forced_deep",
            confidence=conf,
            uncertainty=None if conf is None else 1.0 - conf,
            escalated=False,
        )
        STATS.record(path="deep", escalated=False, unscored=conf is None, truncated=deep.truncated)
        return result

    standard = call(
        client,
        model=cfg.standard_model,
        effort=cfg.standard_effort,
        max_tokens=cfg.standard_max_tokens,
        system=_STANDARD_SYSTEM,
        query=query,
    )
    conf = parse_confidence(standard.text)
    uncertainty = None if conf is None else 1.0 - conf
    truncated = standard.truncated
    unscored = conf is None

    should_escalate = chosen == "auto" and (
        unscored or truncated or (uncertainty is not None and uncertainty >= cfg.threshold)
    )

    if not should_escalate:
        reason = "forced_standard" if chosen == "standard" else "confident_standard"
        if truncated and chosen == "standard":
            reason = "truncated_no_escalate"
        elif unscored and chosen == "standard":
            reason = "unscored_no_escalate"
        result = RouteResult(
            answer=_strip_confidence_line(standard.text),
            model=cfg.standard_model,
            path="standard",
            reason=reason,
            confidence=conf,
            uncertainty=uncertainty,
            escalated=False,
        )
        STATS.record(path="standard", escalated=False, unscored=unscored, truncated=truncated)
        return result

    if unscored:
        escalate_reason = "unscored"
    elif truncated:
        escalate_reason = "truncated"
    else:
        escalate_reason = "uncertainty_above_threshold"

    deep = call(
        client,
        model=cfg.deep_model,
        effort=cfg.deep_effort,
        max_tokens=max_tokens or cfg.deep_max_tokens,
        system=_DEEP_SYSTEM,
        query=query,
    )
    deep_conf = parse_confidence(deep.text)
    result = RouteResult(
        answer=_strip_confidence_line(deep.text),
        model=cfg.deep_model,
        path="deep",
        reason=escalate_reason,
        confidence=deep_conf,
        uncertainty=None if deep_conf is None else 1.0 - deep_conf,
        escalated=True,
    )
    STATS.record(path="deep", escalated=True, unscored=unscored, truncated=truncated)
    return result
