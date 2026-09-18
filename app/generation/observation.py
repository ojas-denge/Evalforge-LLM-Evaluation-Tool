from dataclasses import dataclass, field
from typing import Any

from app.generation.usage import GenerationUsage


@dataclass(frozen=True)
class GenerationObservation:
    """Immutable record of what a provider actually returned."""

    requested_model: str
    actual_model: str | None
    actual_provider: str | None

    http_status: int | None
    finish_reason: str | None

    content: str | None
    refusal: str | None
    reasoning: str | None
    reasoning_details: list[dict[str, Any]]

    usage: GenerationUsage
    latency_ms: float

    response_format_requested: dict[str, Any] | None

    raw_response: dict[str, Any] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)
