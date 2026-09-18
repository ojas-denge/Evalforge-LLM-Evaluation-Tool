from dataclasses import dataclass

from app.evaluation.comparison import EvaluationComparison
from app.evaluation.regression import RegressionPolicy


@dataclass(frozen=True)
class CIGateResult:
    passed: bool
    reasons: list[str]


class EvaluationCIGate:
    def __init__(self, policy: RegressionPolicy) -> None:
        self.policy = policy

    def evaluate(
        self,
        comparison: EvaluationComparison,
    ) -> CIGateResult:
        regression = self.policy.evaluate(comparison)

        return CIGateResult(
            passed=not regression.regression_detected,
            reasons=regression.reasons,
        )
