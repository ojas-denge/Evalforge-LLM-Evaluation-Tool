from dataclasses import dataclass, field

from app.evaluation.comparison import EvaluationComparison


@dataclass(frozen=True)
class RegressionResult:
    regression_detected: bool
    reasons: list[str] = field(default_factory=list)


class RegressionPolicy:
    def __init__(
        self,
        *,
        max_mrr_drop: float = 0.0,
        max_latency_increase_ms: float | None = 0.0,
        max_correctness_drop: float | None = None,
        max_groundedness_drop: float | None = None,
        max_cost_increase_usd: float | None = None,
        max_conflict_rate: float | None = None,
    ) -> None:
        if max_mrr_drop < 0:
            raise ValueError("max_mrr_drop must be non-negative")

        if (
            max_latency_increase_ms is not None
            and max_latency_increase_ms < 0
        ):
            raise ValueError(
                "max_latency_increase_ms must be non-negative"
            )

        if max_correctness_drop is not None and max_correctness_drop < 0:
            raise ValueError("max_correctness_drop must be non-negative")

        if max_groundedness_drop is not None and max_groundedness_drop < 0:
            raise ValueError("max_groundedness_drop must be non-negative")

        if max_cost_increase_usd is not None and max_cost_increase_usd < 0:
            raise ValueError("max_cost_increase_usd must be non-negative")

        if max_conflict_rate is not None and max_conflict_rate < 0:
            raise ValueError("max_conflict_rate must be non-negative")

        self.max_mrr_drop = max_mrr_drop
        self.max_latency_increase_ms = max_latency_increase_ms
        self.max_correctness_drop = max_correctness_drop
        self.max_groundedness_drop = max_groundedness_drop
        self.max_cost_increase_usd = max_cost_increase_usd
        self.max_conflict_rate = max_conflict_rate

    def evaluate(
        self,
        comparison: EvaluationComparison,
    ) -> RegressionResult:
        reasons: list[str] = []

        mrr_delta = comparison.metric_deltas["mean_mrr"]

        if mrr_delta < -self.max_mrr_drop:
            reasons.append("mean_mrr")

        latency_delta = comparison.metric_deltas[
            "mean_retrieval_latency_ms"
        ]

        if (
            self.max_latency_increase_ms is not None
            and latency_delta > self.max_latency_increase_ms
        ):
            reasons.append("mean_retrieval_latency_ms")

        if (
            self.max_correctness_drop is not None
            and comparison.mean_correctness_delta is not None
            and comparison.mean_correctness_delta < -self.max_correctness_drop
        ):
            reasons.append("mean_correctness")

        if (
            self.max_groundedness_drop is not None
            and comparison.mean_groundedness_delta is not None
            and comparison.mean_groundedness_delta < -self.max_groundedness_drop
        ):
            reasons.append("mean_groundedness")

        if (
            self.max_cost_increase_usd is not None
            and comparison.cost_delta_usd is not None
            and comparison.cost_delta_usd > self.max_cost_increase_usd
        ):
            reasons.append("cost_usd")

        if (
            self.max_conflict_rate is not None
            and comparison.conflict_rate_delta is not None
            and comparison.conflict_rate_delta > self.max_conflict_rate
        ):
            reasons.append("conflict_rate")

        return RegressionResult(
            regression_detected=bool(reasons),
            reasons=reasons,
        )
