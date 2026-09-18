from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.generation.usage import GenerationUsage
from app.models.evaluation import AnswerJudgeResult


class JudgeStatus(str, Enum):
    VALID = "valid"
    SYSTEM_ERROR = "system_error"


class JudgeVerdict(BaseModel):
    """Structured judge result with explicit system-error state."""

    result: AnswerJudgeResult | None = None
    status: JudgeStatus
    judge_model: str = Field(min_length=1)
    judge_provider: str = Field(min_length=1)
    judge_latency_ms: float = Field(ge=0.0)
    judge_tokens: GenerationUsage
    judge_cost_usd: float = Field(ge=0.0)
    structured_output_valid: bool
    judge_failure: str | None = Field(default=None)
    judge_metadata: dict[str, Any] = Field(default_factory=dict)
