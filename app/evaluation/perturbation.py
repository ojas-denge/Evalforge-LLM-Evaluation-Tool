from dataclasses import dataclass
from enum import Enum

from app.models.evaluation import AnswerJudgeResult, RetrievedEvidence


class PerturbationType(str, Enum):
    EVIDENCE_REORDER = "evidence_reorder"
    ANSWER_REFERENCE_SWAP = "answer_reference_swap"


@dataclass(frozen=True)
class PerturbationResult:
    case_id: str
    perturbation_type: PerturbationType
    original_correct: bool
    original_grounded: bool
    perturbed_correct: bool
    perturbed_grounded: bool
    is_stable: bool  # True if both verdicts match


def reorder_evidence(
    evidence: list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    """Reverse the order of evidence items (deterministic perturbation)."""
    reversed_evidence = list(reversed(evidence))
    # Re-assign ranks to reflect new order
    return [
        RetrievedEvidence(
            rank=i + 1,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            distance=item.distance,
            text=item.text,
        )
        for i, item in enumerate(reversed_evidence)
    ]


def compute_perturbation_stability(
    results: list[PerturbationResult],
) -> float:
    """Fraction of cases where the verdict is stable under perturbation."""
    if not results:
        raise ValueError("No perturbation results to compute stability")

    stable = sum(1 for r in results if r.is_stable)
    return stable / len(results)
