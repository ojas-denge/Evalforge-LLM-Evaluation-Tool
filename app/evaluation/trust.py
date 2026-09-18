from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TrustTier(str, Enum):
    DETERMINISTIC_FACT = "tier_1"
    DETERMINISTIC_HEURISTIC = "tier_2"
    SEMANTIC_JUDGMENT = "tier_3"
    AGGREGATE = "tier_4"


class ConflictType(str, Enum):
    JUDGE_CONTRADICTS_RETRIEVAL = "judge_contradicts_retrieval"
    JUDGE_CONTRADICTS_COVERAGE = "judge_contradicts_coverage"
    JUDGE_INTERNAL_INCONSISTENCY = "judge_internal_inconsistency"


class TrustAssessment(BaseModel):
    tier_1_signals: dict[str, Any] = Field(default_factory=dict)
    tier_2_signals: dict[str, Any] = Field(default_factory=dict)
    tier_3_signals: dict[str, Any] = Field(default_factory=dict)
    conflicts: list[ConflictType] = Field(default_factory=list)
    final_verdict: str = Field(min_length=1)
    confidence: str = Field(min_length=1)
    system_error: str | None = None
