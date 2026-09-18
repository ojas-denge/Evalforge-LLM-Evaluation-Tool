import uuid
from datetime import datetime, timezone

from app.db.models import EvaluationRunModel
from app.db.repository import EvaluationRepository
from app.db.session import SessionLocal
from app.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
    RetrievedEvidence,
    RetrievalConfig,
)


def build_test_run() -> EvaluationRun:
    return EvaluationRun(
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc),
        dataset_size=1,
        retrieval_config=RetrievalConfig(
            mode="dense",
            top_k=5,
            candidate_k=None,
            reranking_enabled=False,
            reranker_candidate_k=10,
            hybrid_retrieval_enabled=False,
        ),
        results=[
            EvaluationResult(
                case_id="repository_test_001",
                question="Does repository persistence work?",
                expected_documents=["test_document"],
                retrieved_evidence=[
                    RetrievedEvidence(
                        rank=1,
                        chunk_id="test_chunk",
                        document_id="test_document",
                        distance=0.1,
                        text="Repository persistence test evidence.",
                    )
                ],
                failure_type="PASS",
                relevant_documents_found=True,
                first_relevant_rank=1,
                missing_documents=[],
                confounding_documents=[],
                hit_at_1=1.0,
                hit_at_3=1.0,
                hit_at_5=1.0,
                recall_at_1=1.0,
                recall_at_3=1.0,
                recall_at_5=1.0,
                mrr=1.0,
                retrieval_latency_ms=20.0,
            )
        ],
        mean_hit_at_1=1.0,
        mean_hit_at_3=1.0,
        mean_hit_at_5=1.0,
        mean_recall_at_1=1.0,
        mean_recall_at_3=1.0,
        mean_recall_at_5=1.0,
        mean_mrr=1.0,
        mean_retrieval_latency_ms=20.0,
    )


def test_save_and_get_run():
    repository = EvaluationRepository()
    run = build_test_run()

    repository.save_run(run)

    try:
        stored = repository.get_run(run.run_id)

        assert stored is not None
        assert stored.run_id == run.run_id
        assert stored.dataset_size == 1
        assert stored.retrieval_config.mode == "dense"

        assert len(stored.results) == 1

        result = stored.results[0]

        assert result.case_id == "repository_test_001"
        assert result.failure_type == "PASS"
        assert result.hit_at_1 == 1.0

        assert len(result.retrieved_evidence) == 1

        evidence = result.retrieved_evidence[0]

        assert evidence.chunk_id == "test_chunk"
        assert evidence.document_id == "test_document"
        assert evidence.rank == 1
        assert evidence.distance == 0.1

    finally:
        with SessionLocal() as session:
            stored_model = session.get(
                EvaluationRunModel,
                uuid.UUID(run.run_id),
            )

            if stored_model is not None:
                session.delete(stored_model)
                session.commit()


def test_list_runs_contains_saved_run():
    repository = EvaluationRepository()
    run = build_test_run()

    repository.save_run(run)

    try:
        runs = repository.list_runs()

        assert any(
            stored_run.run_id == run.run_id
            for stored_run in runs
        )

    finally:
        with SessionLocal() as session:
            stored_model = session.get(
                EvaluationRunModel,
                uuid.UUID(run.run_id),
            )

            if stored_model is not None:
                session.delete(stored_model)
                session.commit()
