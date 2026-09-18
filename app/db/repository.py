from uuid import UUID

from sqlalchemy import select

from app.db.models import (
    EvaluationResultModel,
    EvaluationRunModel,
    RetrievedEvidenceModel,
)
from app.db.session import SessionLocal
from app.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
    RetrievedEvidence,
    RetrievalConfig,
)


class EvaluationRepository:
    def save_run(self, run: EvaluationRun) -> None:
        with SessionLocal() as session:
            run_model = EvaluationRunModel(
                run_id=UUID(run.run_id),
                created_at=run.created_at,
                dataset_size=run.dataset_size,
                retrieval_mode=run.retrieval_config.mode,
                top_k=run.retrieval_config.top_k,
                candidate_k=run.retrieval_config.candidate_k,
                reranking_enabled=run.retrieval_config.reranking_enabled,
                reranker_candidate_k=run.retrieval_config.reranker_candidate_k,
                hybrid_retrieval_enabled=(
                    run.retrieval_config.hybrid_retrieval_enabled
                ),
                mean_hit_at_1=run.mean_hit_at_1,
                mean_hit_at_3=run.mean_hit_at_3,
                mean_hit_at_5=run.mean_hit_at_5,
                mean_recall_at_1=run.mean_recall_at_1,
                mean_recall_at_3=run.mean_recall_at_3,
                mean_recall_at_5=run.mean_recall_at_5,
                mean_mrr=run.mean_mrr,
                mean_retrieval_latency_ms=(
                    run.mean_retrieval_latency_ms
                ),
            )

            for result in run.results:
                result_model = EvaluationResultModel(
                    case_id=result.case_id,
                    question=result.question,
                    expected_documents=result.expected_documents,
                    failure_type=result.failure_type,
                    relevant_documents_found=(
                        result.relevant_documents_found
                    ),
                    first_relevant_rank=result.first_relevant_rank,
                    missing_documents=result.missing_documents,
                    confounding_documents=result.confounding_documents,
                    hit_at_1=result.hit_at_1,
                    hit_at_3=result.hit_at_3,
                    hit_at_5=result.hit_at_5,
                    recall_at_1=result.recall_at_1,
                    recall_at_3=result.recall_at_3,
                    recall_at_5=result.recall_at_5,
                    mrr=result.mrr,
                    retrieval_latency_ms=result.retrieval_latency_ms,
                )

                for evidence in result.retrieved_evidence:
                    result_model.evidence.append(
                        RetrievedEvidenceModel(
                            rank=evidence.rank,
                            chunk_id=evidence.chunk_id,
                            document_id=evidence.document_id,
                            distance=evidence.distance,
                            text=evidence.text,
                        )
                    )

                run_model.results.append(result_model)

            session.add(run_model)
            session.commit()

    def get_run(self, run_id: str) -> EvaluationRun | None:
        with SessionLocal() as session:
            run_model = session.get(
                EvaluationRunModel,
                UUID(run_id),
            )

            if run_model is None:
                return None

            return self._to_domain_model(run_model)

    def list_runs(self) -> list[EvaluationRun]:
        with SessionLocal() as session:
            statement = (
                select(EvaluationRunModel)
                .order_by(EvaluationRunModel.created_at.desc())
            )

            runs = session.scalars(statement).all()

            return [
                self._to_domain_model(run)
                for run in runs
            ]

    def _to_domain_model(
        self,
        run_model: EvaluationRunModel,
    ) -> EvaluationRun:
        results = [
            EvaluationResult(
                case_id=result.case_id,
                question=result.question,
                expected_documents=result.expected_documents,
                retrieved_evidence=[
                    RetrievedEvidence(
                        rank=evidence.rank,
                        chunk_id=evidence.chunk_id,
                        document_id=evidence.document_id,
                        distance=evidence.distance,
                        text=evidence.text,
                    )
                    for evidence in result.evidence
                ],
                failure_type=result.failure_type,
                relevant_documents_found=(
                    result.relevant_documents_found
                ),
                first_relevant_rank=result.first_relevant_rank,
                missing_documents=result.missing_documents,
                confounding_documents=result.confounding_documents,
                hit_at_1=result.hit_at_1,
                hit_at_3=result.hit_at_3,
                hit_at_5=result.hit_at_5,
                recall_at_1=result.recall_at_1,
                recall_at_3=result.recall_at_3,
                recall_at_5=result.recall_at_5,
                mrr=result.mrr,
                retrieval_latency_ms=result.retrieval_latency_ms,
            )
            for result in run_model.results
        ]

        retrieval_config = RetrievalConfig(
            mode=run_model.retrieval_mode,
            top_k=run_model.top_k,
            candidate_k=run_model.candidate_k,
            reranking_enabled=run_model.reranking_enabled,
            reranker_candidate_k=run_model.reranker_candidate_k,
            hybrid_retrieval_enabled=(
                run_model.hybrid_retrieval_enabled
            ),
        )

        return EvaluationRun(
            run_id=str(run_model.run_id),
            created_at=run_model.created_at,
            dataset_size=run_model.dataset_size,
            retrieval_config=retrieval_config,
            results=results,
            mean_hit_at_1=run_model.mean_hit_at_1,
            mean_hit_at_3=run_model.mean_hit_at_3,
            mean_hit_at_5=run_model.mean_hit_at_5,
            mean_recall_at_1=run_model.mean_recall_at_1,
            mean_recall_at_3=run_model.mean_recall_at_3,
            mean_recall_at_5=run_model.mean_recall_at_5,
            mean_mrr=run_model.mean_mrr,
            mean_retrieval_latency_ms=(
                run_model.mean_retrieval_latency_ms
            ),
        )
