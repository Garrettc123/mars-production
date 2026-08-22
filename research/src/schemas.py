from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class PredictionRecord:
    item_id: str
    condition: str
    model_id: str
    prompt_version: str
    seed: int
    predicted_answer: Any
    is_correct: Optional[bool]
    confidence: float
    uncertainty: float
    routed_deep: bool
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    verifier: Optional[str] = None
    timestamp_utc: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
