from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# G3B is intentionally built from the existing EvalForge corpus.
# This script creates a controlled HARD-CASE DATASET SPECIFICATION.
# It does not call an LLM and does not modify production evaluation code.

FORENSIC_PATH = Path("logs/dense_vs_crossencoder_forensic.json")
DATASET_PATH = Path("data/evaluation_cases.json")
OUTPUT_PATH = Path("logs/g3b_dataset.json")


CATEGORIES = {
    "multi_evidence": 6,
    "distractor_competition": 6,
    "ranking_sensitive": 6,
    "contradiction": 6,
    "missing_evidence": 6,
}


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_cases(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("cases", "items", "data", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    return []


def get_case_id(case: dict[str, Any]) -> str | None:
    for key in ("case_id", "id", "caseId"):
        if case.get(key):
            return str(case[key])
    return None


def get_expected_docs(case: dict[str, Any]) -> list[str]:
    value = (
        case.get("expected_documents")
        or case.get("expected_docs")
        or case.get("gold_documents")
        or []
    )
    return [str(x) for x in value]


def get_question(case: dict[str, Any]) -> str:
    return str(case.get("question") or "")


def get_expected_answer(case: dict[str, Any]) -> str:
    return str(case.get("expected_answer") or case.get("answer") or "")


def get_topics(case: dict[str, Any]) -> list[str]:
    value = case.get("expected_topics") or case.get("topics") or []
    return [str(x) for x in value]


def classify_existing_cases(
    cases: list[dict[str, Any]],
    forensic: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    forensic_map = {
        x.get("case_id"): x
        for x in forensic.get("cases", [])
        if x.get("case_id")
    }

    buckets = {key: [] for key in CATEGORIES}

    # The purpose here is NOT to pretend we can infer a perfect adversarial
    # label automatically. We use conservative signals from the existing
    # corpus to identify candidates for manual G3B construction.
    for case in cases:
        cid = get_case_id(case)
        if not cid:
            continue

        f = forensic_map.get(cid, {})
        expected = get_expected_docs(case)
        classification = f.get("classification")

        if len(expected) >= 2:
            buckets["multi_evidence"].append(case)

        if classification in {
            "RANKING_FAILURE",
            "MULTI_EVIDENCE_RANKING_FAILURE",
        }:
            buckets["ranking_sensitive"].append(case)

        if classification == "RETRIEVAL_FAILURE":
            buckets["missing_evidence"].append(case)

    # Existing corpus may not contain enough cases for all five categories.
    # That is intentional: missing slots are emitted as construction slots
    # rather than fabricated questions/answers.
    return buckets


def make_slot(category: str, index: int, source_case: dict[str, Any] | None):
    slot = {
        "case_id": f"g3b_{category[:4]}_{index:03d}",
        "category": category,
        "status": "SOURCE_CANDIDATE" if source_case else "CONSTRUCTION_REQUIRED",
        "question": get_question(source_case) if source_case else None,
        "expected_answer": get_expected_answer(source_case) if source_case else None,
        "expected_topics": get_topics(source_case) if source_case else [],
        "expected_documents": get_expected_docs(source_case) if source_case else [],
        "difficulty_reason": None,
        "construction_requirements": [],
        "source_case_id": get_case_id(source_case) if source_case else None,
    }

    requirements = {
        "multi_evidence": [
            "Answer must require information from at least two distinct documents.",
            "No single expected document should contain the complete answer.",
        ],
        "distractor_competition": [
            "At least one highly similar but non-sufficient distractor must exist.",
            "Correct evidence must remain distinguishable from the distractor.",
        ],
        "ranking_sensitive": [
            "Correct evidence should be retrievable in the candidate pool.",
            "Its rank should plausibly change between Dense and CrossEncoder.",
        ],
        "contradiction": [
            "Use related policies/rules with different applicability rather than invented contradictions.",
            "Question must require selecting the applicable evidence.",
        ],
        "missing_evidence": [
            "Required evidence must genuinely be absent from the retrieved context.",
            "Expected behavior must be evidence-aware rather than hallucinated.",
        ],
    }

    slot["construction_requirements"] = requirements[category]
    slot["difficulty_reason"] = {
        "multi_evidence": "Tests cross-document synthesis.",
        "distractor_competition": "Tests semantic distractor resistance.",
        "ranking_sensitive": "Tests whether ranking changes final answer quality.",
        "contradiction": "Tests evidence applicability under competing statements.",
        "missing_evidence": "Tests grounded behavior when required evidence is unavailable.",
    }[category]

    return slot


def main():
    forensic = load(FORENSIC_PATH)
    raw_dataset = load(DATASET_PATH)
    cases = flatten_cases(raw_dataset)

    buckets = classify_existing_cases(cases, forensic)

    output_cases = []
    selection_summary = {}

    for category, target_count in CATEGORIES.items():
        candidates = buckets.get(category, [])
        selection_summary[category] = {
            "target": target_count,
            "existing_candidates": len(candidates),
        }

        for i in range(1, target_count + 1):
            source = candidates[i - 1] if i <= len(candidates) else None
            output_cases.append(make_slot(category, i, source))

    output = {
        "experiment": "g3b_hard_case_dataset",
        "version": "0.1",
        "status": "DESIGN_STAGE",
        "dataset_size_target": sum(CATEGORIES.values()),
        "source_corpus": str(DATASET_PATH),
        "source_forensics": str(FORENSIC_PATH),
        "api_calls_used": 0,
        "selection_summary": selection_summary,
        "design_rules": {
            "fixed_corpus": True,
            "production_pipeline_modified": False,
            "llm_generation_during_build": False,
            "retrieval_screening_after_dataset_completion": True,
        },
        "categories": CATEGORIES,
        "cases": output_cases,
        "next_step": (
            "Manually finalize CONSTRUCTION_REQUIRED slots from the existing "
            "corpus, then run retrieval-only Dense vs CrossEncoder screening "
            "before any generation/judging calls."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 72)
    print("G3B — HARD-CASE DATASET DESIGN")
    print("=" * 72)
    print(f"Target cases: {output['dataset_size_target']}")
    print("API calls: 0")
    print()
    for category, summary in selection_summary.items():
        print(
            f"{category:24s} "
            f"target={summary['target']} "
            f"existing_candidates={summary['existing_candidates']}"
        )
    print()
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
