from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.repository import EvaluationRepository
from app.evaluation.comparison import compare_runs
from app.evaluation.regression import RegressionPolicy
from app.generation.base import GenerationRequest, Generator
from app.generation.factory import create_generator
from app.models.schemas import QueryRequest, QueryResponse
from app.observability.tracing import Tracer
from app.retrieval.retriever import Retriever


settings = get_settings()

configure_logging(settings.log_level)
logger = get_logger(__name__)

tracer = Tracer()
retriever = Retriever(tracer=tracer)
generator: Generator = create_generator(settings, tracer=tracer)
repository = EvaluationRepository()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "evalforge",
        "version": settings.app_version,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    request_start = perf_counter()
    request_id = str(uuid4())

    with tracer.trace(
        name="rag_request",
        input={"question": tracer.redact(request.question)},
        metadata={
            "endpoint": "/query",
            "request_id": request_id,
            "retrieval_top_k": 3,
        },
    ) as observation:

        logger.info("Query received: %s", request.question)

        retrieval = retriever.retrieve(
            query=request.question,
            top_k=3,
            trace_metadata={"request_id": request_id},
        )

        citations = [
            result.document_id
            for result in retrieval.results
        ]

        generation_request = GenerationRequest(
            question=request.question,
            context=retrieval.results,
            model=settings.llm_model,
            temperature=0.0,
            metadata={"request_id": request_id},
        )

        generation = generator.generate(generation_request)

        total_latency_ms = (
            perf_counter() - request_start
        ) * 1000

        response = QueryResponse(
            answer=generation.answer,
            citations=citations,
            confidence=None,
            latency_ms=total_latency_ms,
            input_tokens=generation.usage.input_tokens,
            output_tokens=generation.usage.output_tokens,
            estimated_cost_usd=generation.estimated_cost_usd,
        )

        if observation is not None:
            observation.update(
                output={
                    "citations": citations,
                    "retrieval_result_count": len(
                        retrieval.results
                    ),
                    "retrieval_latency_ms": retrieval.latency_ms,
                    "generation_latency_ms": generation.latency_ms,
                    "input_tokens": generation.usage.input_tokens,
                    "output_tokens": generation.usage.output_tokens,
                    "total_tokens": generation.usage.total_tokens,
                    "estimated_cost_usd": generation.estimated_cost_usd,
                    "total_latency_ms": total_latency_ms,
                }
            )

        return response


@app.get("/evaluations")
def list_evaluations():
    return repository.list_runs()


@app.get("/evaluations/{run_id}")
def get_evaluation(run_id):
    run = repository.get_run(run_id)

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation run not found: {run_id}",
        )

    return run


@app.get(
    "/evaluations/{baseline_run_id}/compare/{candidate_run_id}"
)
def compare_evaluation_runs(
    baseline_run_id,
    candidate_run_id,
):
    baseline = repository.get_run(baseline_run_id)
    candidate = repository.get_run(candidate_run_id)

    if baseline is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Baseline evaluation run not found: "
                f"{baseline_run_id}"
            ),
        )

    if candidate is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Candidate evaluation run not found: "
                f"{candidate_run_id}"
            ),
        )

    comparison = compare_runs(
        baseline,
        candidate,
    )

    policy = RegressionPolicy(
        max_mrr_drop=0.02,
        max_latency_increase_ms=100.0,
    )

    regression = policy.evaluate(comparison)

    return {
        "baseline_run_id": comparison.baseline_run_id,
        "candidate_run_id": comparison.candidate_run_id,
        "metric_deltas": comparison.metric_deltas,
        "case_changes": [
            {
                "case_id": change.case_id,
                "baseline_failure_type": (
                    change.baseline_failure_type
                ),
                "candidate_failure_type": (
                    change.candidate_failure_type
                ),
                "baseline_first_relevant_rank": (
                    change.baseline_first_relevant_rank
                ),
                "candidate_first_relevant_rank": (
                    change.candidate_first_relevant_rank
                ),
                "status": change.status,
            }
            for change in comparison.case_changes
        ],
        "regression": {
            "regression_detected": (
                regression.regression_detected
            ),
            "reasons": regression.reasons,
        },
    }
