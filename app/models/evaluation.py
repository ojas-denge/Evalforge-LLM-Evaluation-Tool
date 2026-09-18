from __future__ import annotations
from datetime import datetime

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, StrictBool

if TYPE_CHECKING:
    from app.evaluation.judge_result import JudgeVerdict


class JudgeConfig(BaseModel):
    model: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0, default=0.0)
    batch_size: int | None = Field(default=None, ge=1)


class RetrievalConfig(BaseModel):
    mode: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    candidate_k: int | None = Field(default=None, ge=1)
    reranking_enabled: bool
    reranker_candidate_k: int | None = Field(default=None, ge=1)
    hybrid_retrieval_enabled: bool


class EvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    expected_documents: list[str] = Field(min_length=1)
    expected_topics: list[str] = Field(default_factory=list)
    category: str = Field(min_length=1)
    difficulty: str = Field(min_length=1)


class RetrievedEvidence(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    distance: float | None = Field(default=None, ge=0.0)
    text: str = Field(min_length=1)


class AnswerJudgeResult(BaseModel):
    answer_correct: StrictBool
    answer_grounded: StrictBool
    topics_covered: list[str] = Field(default_factory=list)
    topics_missing: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)


class EvaluationResult(BaseModel):
    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_documents: list[str] = Field(min_length=1)

    generated_answer: str | None = Field(default=None)
    topic_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    answer_judge: AnswerJudgeResult | None = Field(default=None)
    judge_verdict: JudgeVerdict | None = Field(default=None)

    retrieved_evidence: list[RetrievedEvidence] = Field(
        default_factory=list
    )

    failure_type: str = Field(min_length=1)
    relevant_documents_found: bool
    first_relevant_rank: int | None = Field(default=None, ge=1)
    missing_documents: list[str] = Field(default_factory=list)
    confounding_documents: list[str] = Field(default_factory=list)

    hit_at_1: float = Field(ge=0.0, le=1.0)
    hit_at_3: float = Field(ge=0.0, le=1.0)
    hit_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_3: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    retrieval_latency_ms: float = Field(ge=0.0)


class EvaluationRun(BaseModel):
    run_id: str = Field(min_length=1)
    created_at: datetime
    dataset_size: int = Field(ge=0)
    retrieval_config: RetrievalConfig
    judge_config: JudgeConfig | None = Field(default=None)
    results: list[EvaluationResult] = Field(default_factory=list)

    mean_hit_at_1: float = Field(ge=0.0, le=1.0)
    mean_hit_at_3: float = Field(ge=0.0, le=1.0)
    mean_hit_at_5: float = Field(ge=0.0, le=1.0)
    mean_recall_at_1: float = Field(ge=0.0, le=1.0)
    mean_recall_at_3: float = Field(ge=0.0, le=1.0)
    mean_recall_at_5: float = Field(ge=0.0, le=1.0)
    mean_mrr: float = Field(ge=0.0, le=1.0)
    mean_topic_coverage: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    mean_retrieval_latency_ms: float = Field(ge=0.0)

from app.evaluation.judge_result import JudgeVerdict
EvaluationResult.model_rebuild()
