from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.generation.observation import GenerationObservation
from app.generation.usage import GenerationUsage
from app.retrieval.retriever import RetrievedDocument


class StructuredOutputStatus(str, Enum):
    """Observed state of requested structured output."""

    NOT_REQUESTED = "not_requested"
    PARSED = "parsed"
    MISSING_CONTENT = "missing_content"
    PARSE_FAILED = "parse_failed"


@dataclass(frozen=True)
class GenerationRequest:
    question: str
    context: list[RetrievedDocument]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    system_prompt: str | None = None
    response_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Provider generation result preserving observed response state."""

    answer: str | None
    model: str
    provider: str
    usage: GenerationUsage
    estimated_cost_usd: float
    latency_ms: float
    finish_reason: str | None = None
    structured_output: dict[str, Any] | None = None
    structured_output_status: StructuredOutputStatus = (
        StructuredOutputStatus.NOT_REQUESTED
    )
    observation: GenerationObservation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Generator(ABC):
    """Provider-independent interface for LLM generation."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Generate a response from a provider."""
        raise NotImplementedError
