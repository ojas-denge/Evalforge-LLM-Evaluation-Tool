from datetime import datetime, timezone
from uuid import uuid4

from app.core.config import get_settings
from app.db.repository import EvaluationRepository
from app.evaluation import EvaluationDataset
from app.evaluation.answer_judge import AnswerJudge
from app.evaluation.diagnostics import analyze_retrieval
from app.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    topic_coverage,
)
from app.generation.base import GenerationRequest, Generator
from app.generation.factory import create_generator
from app.models.evaluation import (
    EvaluationResult,
    EvaluationRun,
    RetrievedEvidence,
    RetrievalConfig,
    JudgeConfig,
)
from app.observability.tracing import Tracer
from app.retrieval.retriever import Retriever


class Evaluator:
    def __init__(
        self,
        retriever=None,
        generator=None,
        tracer=None,
        repository=None,
        answer_judge: AnswerJudge | None = None,
    ):
        self.tracer = tracer or Tracer()
        self.retriever = retriever or Retriever(tracer=self.tracer)
        self.generator = generator or create_generator(
            settings=get_settings(),
            tracer=self.tracer,
        )
        self.repository = repository or EvaluationRepository()
        self.answer_judge = answer_judge

    def _build_retrieval_config(self):
        return RetrievalConfig(
            mode=self.retriever.mode,
            top_k=5,
            candidate_k=self.retriever.candidate_k,
            reranking_enabled=self.retriever.reranking_enabled,
            reranker_candidate_k=self.retriever.reranker_candidate_k,
            hybrid_retrieval_enabled=self.retriever.hybrid_retrieval_enabled,
        )

    def evaluate_case(self, case):
        retrieval = self.retriever.retrieve(
            query=case.question,
            top_k=5,
        )

        retrieved_evidence = [
            RetrievedEvidence(
                rank=result.rank,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                distance=result.distance,
                text=result.text,
            )
            for result in retrieval.results
        ]

        retrieved_documents = [
            evidence.document_id
            for evidence in retrieved_evidence
        ]

        diagnostic = analyze_retrieval(
            case.expected_documents,
            retrieved_documents,
        )

        generation_request = GenerationRequest(
            question=case.question,
            context=retrieval.results,
            model=None,
            temperature=0.0,
        )

        generation = self.generator.generate(generation_request)

        case_topic_coverage = None

        if case.expected_topics:
            case_topic_coverage = topic_coverage(
                case.expected_topics,
                generation.answer,
            )

        judged_answer = None
        judge_verdict = None

        if self.answer_judge is not None:
            verdict = self.answer_judge.judge(
                question=case.question,
                expected_answer=case.expected_answer,
                expected_topics=case.expected_topics,
                generated_answer=generation.answer,
                retrieved_evidence=retrieved_evidence,
            )
            judge_verdict = verdict
            judged_answer = verdict.result

        return EvaluationResult(
            case_id=case.case_id,
            question=case.question,
            expected_documents=case.expected_documents,
            generated_answer=generation.answer,
            topic_coverage=case_topic_coverage,
            answer_judge=judged_answer,
            judge_verdict=judge_verdict,
            retrieved_evidence=retrieved_evidence,
            failure_type=diagnostic.failure_type,
            relevant_documents_found=diagnostic.relevant_documents_found,
            first_relevant_rank=diagnostic.first_relevant_rank,
            missing_documents=diagnostic.missing_documents,
            confounding_documents=diagnostic.confounding_documents,
            hit_at_1=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                1,
            ),
            hit_at_3=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                3,
            ),
            hit_at_5=hit_at_k(
                case.expected_documents,
                retrieved_documents,
                5,
            ),
            recall_at_1=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                1,
            ),
            recall_at_3=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                3,
            ),
            recall_at_5=recall_at_k(
                case.expected_documents,
                retrieved_documents,
                5,
            ),
            mrr=reciprocal_rank(
                case.expected_documents,
                retrieved_documents,
            ),
            retrieval_latency_ms=retrieval.latency_ms,
        )

    def evaluate_dataset(self, dataset):
        with self.tracer.trace(
            name="evaluation_run",
            input={"dataset_size": len(dataset)},
            metadata={
                "retriever_mode": self.retriever.mode,
                "retrieval_top_k": 5,
                "answer_judge_enabled": self.answer_judge is not None,
            },
        ) as observation:
            run = self._evaluate_dataset(dataset)

            self.repository.save_run(run)

            if observation is not None:
                observation.update(
                    output={
                        "dataset_size": run.dataset_size,
                        "mean_hit_at_1": run.mean_hit_at_1,
                        "mean_hit_at_3": run.mean_hit_at_3,
                        "mean_hit_at_5": run.mean_hit_at_5,
                        "mean_recall_at_1": run.mean_recall_at_1,
                        "mean_recall_at_3": run.mean_recall_at_3,
                        "mean_recall_at_5": run.mean_recall_at_5,
                        "mean_mrr": run.mean_mrr,
                        "mean_topic_coverage": (
                            run.mean_topic_coverage
                        ),
                    }
                )

            return run

    def _evaluate_dataset(self, dataset):
        results = [
            self.evaluate_case(case)
            for case in dataset.cases
        ]

        if results:
            mean_hit_at_1 = sum(
                result.hit_at_1 for result in results
            ) / len(results)

            mean_hit_at_3 = sum(
                result.hit_at_3 for result in results
            ) / len(results)

            mean_hit_at_5 = sum(
                result.hit_at_5 for result in results
            ) / len(results)

            mean_recall_at_1 = sum(
                result.recall_at_1 for result in results
            ) / len(results)

            mean_recall_at_3 = sum(
                result.recall_at_3 for result in results
            ) / len(results)

            mean_recall_at_5 = sum(
                result.recall_at_5 for result in results
            ) / len(results)

            mean_mrr = sum(
                result.mrr for result in results
            ) / len(results)

            topic_scores = [
                result.topic_coverage
                for result in results
                if result.topic_coverage is not None
            ]

            mean_topic_coverage = (
                sum(topic_scores) / len(topic_scores)
                if topic_scores
                else None
            )

            mean_retrieval_latency_ms = sum(
                result.retrieval_latency_ms
                for result in results
            ) / len(results)

        else:
            mean_hit_at_1 = 0.0
            mean_hit_at_3 = 0.0
            mean_hit_at_5 = 0.0
            mean_recall_at_1 = 0.0
            mean_recall_at_3 = 0.0
            mean_recall_at_5 = 0.0
            mean_mrr = 0.0
            mean_topic_coverage = None
            mean_retrieval_latency_ms = 0.0

        judge_config = None
        if self.answer_judge is not None:
            gen = getattr(self.answer_judge, 'generator', None)
            if gen and getattr(gen, 'model', None) and getattr(gen, 'provider', None):
                judge_config = JudgeConfig(
                    model=gen.model,
                    provider=gen.provider,
                    temperature=getattr(gen, 'temperature', 0.0),
                    batch_size=None
                )

        return EvaluationRun(
            run_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
            dataset_size=len(dataset),
            retrieval_config=self._build_retrieval_config(),
            judge_config=judge_config,
            results=results,
            mean_hit_at_1=mean_hit_at_1,
            mean_hit_at_3=mean_hit_at_3,
            mean_hit_at_5=mean_hit_at_5,
            mean_recall_at_1=mean_recall_at_1,
            mean_recall_at_3=mean_recall_at_3,
            mean_recall_at_5=mean_recall_at_5,
            mean_mrr=mean_mrr,
            mean_topic_coverage=mean_topic_coverage,
            mean_retrieval_latency_ms=mean_retrieval_latency_ms,
        )
