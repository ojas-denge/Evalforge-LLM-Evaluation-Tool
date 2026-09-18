from app.evaluation.judge_result import JudgeStatus
from app.evaluation.trust import ConflictType, TrustAssessment
from app.models.evaluation import EvaluationResult


class ConflictDetector:
    def __init__(
        self,
        *,
        topic_coverage_threshold: float = 0.3,
    ) -> None:
        self.topic_coverage_threshold = topic_coverage_threshold

    def assess(self, result: EvaluationResult) -> TrustAssessment:
        # Tier 1: deterministic facts
        tier_1 = {
            "failure_type": result.failure_type,
            "relevant_documents_found": result.relevant_documents_found,
            "first_relevant_rank": result.first_relevant_rank,
            "hit_at_1": result.hit_at_1,
            "mrr": result.mrr,
            "retrieval_latency_ms": result.retrieval_latency_ms,
        }

        # Tier 2: deterministic heuristics
        tier_2 = {
            "topic_coverage": result.topic_coverage,
            "diagnostic_classification": result.failure_type,
        }

        # Tier 3: semantic judgment.
        # A system-error judge has no semantic result and must not
        # participate in conflict detection.
        judge_verdict = result.judge_verdict
        judge_result = (
            judge_verdict.result
            if judge_verdict is not None
            else result.answer_judge
        )

        tier_3: dict = {}
        if judge_result is not None:
            tier_3 = {
                "answer_correct": judge_result.answer_correct,
                "answer_grounded": judge_result.answer_grounded,
                "topics_covered_count": len(judge_result.topics_covered),
                "topics_missing_count": len(judge_result.topics_missing),
                "unsupported_claims_count": len(
                    judge_result.unsupported_claims
                ),
            }

        system_error = None
        if (
            judge_verdict is not None
            and judge_verdict.status == JudgeStatus.SYSTEM_ERROR
        ):
            system_error = judge_verdict.judge_failure

        conflicts = self._detect_conflicts(
            result,
            judge_verdict=judge_verdict,
            judge_result=judge_result,
        )

        final_verdict, confidence = self._determine_verdict(
            result,
            conflicts,
            judge_verdict=judge_verdict,
            judge_result=judge_result,
        )

        return TrustAssessment(
            tier_1_signals=tier_1,
            tier_2_signals=tier_2,
            tier_3_signals=tier_3,
            conflicts=conflicts,
            final_verdict=final_verdict,
            confidence=confidence,
            system_error=system_error,
        )

    def _detect_conflicts(
        self,
        result: EvaluationResult,
        *,
        judge_verdict=None,
        judge_result=None,
    ) -> list[ConflictType]:
        conflicts: list[ConflictType] = []

        # A system-error judge has no semantic result.
        # It is not a conflict and must not be treated as a semantic failure.
        if (
            judge_verdict is not None
            and judge_verdict.status == JudgeStatus.SYSTEM_ERROR
        ):
            return conflicts

        if judge_result is None:
            return conflicts

        # A deterministic failure cannot be overridden by a semantic judge.
        if (
            result.failure_type != "PASS"
            and judge_result.answer_correct
            and judge_result.answer_grounded
        ):
            conflicts.append(ConflictType.JUDGE_CONTRADICTS_RETRIEVAL)

        if (
            judge_result.answer_correct
            and result.topic_coverage is not None
            and result.topic_coverage < self.topic_coverage_threshold
        ):
            conflicts.append(ConflictType.JUDGE_CONTRADICTS_COVERAGE)

        if (
            judge_result.answer_grounded
            and len(judge_result.unsupported_claims) > 0
        ):
            conflicts.append(ConflictType.JUDGE_INTERNAL_INCONSISTENCY)

        return conflicts

    def _determine_verdict(
        self,
        result: EvaluationResult,
        conflicts: list[ConflictType],
        *,
        judge_verdict=None,
        judge_result=None,
    ) -> tuple[str, str]:
        # Evaluation infrastructure failed.
        # Do not convert the missing judge decision into PASS/FAIL.
        if (
            judge_verdict is not None
            and judge_verdict.status == JudgeStatus.SYSTEM_ERROR
        ):
            return "SYSTEM_ERROR", "LOW"

        # Conflicting signals require review.
        if conflicts:
            return "REVIEW", "LOW"

        # Deterministic facts remain authoritative.
        # The judge cannot promote a deterministic failure to PASS.
        if result.failure_type != "PASS":
            return "FAIL", "MEDIUM"

        # No judge means we only have deterministic evaluation.
        if judge_result is None:
            return "PASS", "MEDIUM"

        if judge_result.answer_correct and judge_result.answer_grounded:
            return "PASS", "HIGH"

        return "FAIL", "HIGH"
