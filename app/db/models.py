import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    dataset_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    retrieval_mode: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    top_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    candidate_k: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    reranking_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    reranker_candidate_k: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    hybrid_retrieval_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    mean_hit_at_1: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_hit_at_3: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_hit_at_5: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_recall_at_1: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_recall_at_3: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_recall_at_5: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_mrr: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mean_retrieval_latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    results: Mapped[list["EvaluationResultModel"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
    )


class EvaluationResultModel(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    case_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    expected_documents: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
    )

    failure_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    relevant_documents_found: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
    )

    first_relevant_rank: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    missing_documents: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
    )

    confounding_documents: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
    )

    hit_at_1: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    hit_at_3: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    hit_at_5: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recall_at_1: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recall_at_3: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    recall_at_5: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    mrr: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    retrieval_latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    run: Mapped["EvaluationRunModel"] = relationship(
        back_populates="results",
    )

    evidence: Mapped[list["RetrievedEvidenceModel"]] = relationship(
        back_populates="result",
        cascade="all, delete-orphan",
    )


class RetrievedEvidenceModel(Base):
    __tablename__ = "retrieved_evidence"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    result_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("evaluation_results.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    chunk_id: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    document_id: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    distance: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    result: Mapped["EvaluationResultModel"] = relationship(
        back_populates="evidence",
    )
