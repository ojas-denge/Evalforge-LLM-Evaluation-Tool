from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class RetrievalDiagnostic:
    failure_type: str
    relevant_documents_found: bool
    first_relevant_rank: int | None
    missing_documents: list[str]
    confounding_documents: list[str]


def analyze_retrieval(
    expected_documents: Sequence[str],
    retrieved_documents: Sequence[str],
) -> RetrievalDiagnostic:
    if not expected_documents:
        raise ValueError("expected_documents cannot be empty")

    expected = set(expected_documents)
    retrieved = list(retrieved_documents)

    relevant_ranks = [
        rank
        for rank, document in enumerate(retrieved, start=1)
        if document in expected
    ]

    found_documents = set(retrieved) & expected
    missing_documents = [
        document
        for document in expected_documents
        if document not in found_documents
    ]

    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None

    confounding_documents = [
        document
        for document in retrieved
        if document not in expected
    ]

    if not relevant_ranks:
        failure_type = "RETRIEVAL_FAILURE"
    elif missing_documents:
        failure_type = "PARTIAL_RECALL"
    elif first_relevant_rank != 1:
        if len(expected) > 1:
            failure_type = "MULTI_EVIDENCE_RANKING_FAILURE"
        else:
            failure_type = "RANKING_FAILURE"
    else:
        failure_type = "PASS"

    return RetrievalDiagnostic(
        failure_type=failure_type,
        relevant_documents_found=bool(relevant_ranks),
        first_relevant_rank=first_relevant_rank,
        missing_documents=missing_documents,
        confounding_documents=confounding_documents,
    )
