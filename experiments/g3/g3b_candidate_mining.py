from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(".")
EVAL_CASES_PATH = ROOT / "data" / "evaluation_cases.json"
FORENSIC_PATH = ROOT / "logs" / "dense_vs_crossencoder_forensic.json"
G1_PATH = ROOT / "logs" / "g1_judged.json"
G2_PATH = ROOT / "logs" / "g2_judged.json"

OUTPUT_PATH = ROOT / "logs" / "g3b_candidate_mining.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def case_list(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        for key in ("cases", "items", "data", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    return []


def case_map(data: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for case in case_list(data):
        cid = case.get("case_id") or case.get("id")
        if cid:
            result[str(cid)] = case
    return result


def forensic_cases(data: Any) -> list[dict[str, Any]]:
    value = data.get("cases", []) if isinstance(data, dict) else []
    return [x for x in value if isinstance(x, dict)]


def docs_from_condition(condition: dict[str, Any]) -> list[str]:
    for key in ("documents", "final_documents"):
        value = condition.get(key)
        if isinstance(value, list):
            return [str(x) for x in value]

    value = condition.get("results")
    if isinstance(value, list):
        docs = []
        for item in value:
            if isinstance(item, dict):
                doc = item.get("document_id")
                if doc:
                    docs.append(str(doc))
        return docs

    return []


def safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def get_answer_state(case: dict[str, Any], condition: str) -> bool | None:
    states = case.get("answer_states")
    if isinstance(states, dict):
        state = states.get(condition)
        if isinstance(state, dict):
            return safe_bool(state.get("answer_correct"))

    answer_correctness = case.get("answer_correctness")
    if isinstance(answer_correctness, dict):
        return safe_bool(answer_correctness.get(condition))

    return None


def get_g1_case_map(data: Any) -> dict[str, dict[str, Any]]:
    # g1_judged.json can be represented in several experiment versions.
    # Search recursively for objects carrying case_id and answer correctness.
    found: dict[str, dict[str, Any]] = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            cid = obj.get("case_id")
            if cid:
                found[str(cid)] = obj
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(data)
    return found


def extract_g1_correct(case: dict[str, Any], condition: str) -> bool | None:
    # Direct forms.
    value = get_answer_state(case, condition)
    if value is not None:
        return value

    # Common nested forms.
    for key in ("judge", "verdict", "result", "answer_judge"):
        obj = case.get(key)
        if isinstance(obj, dict):
            value = safe_bool(obj.get("answer_correct"))
            if value is not None:
                return value

    return None


def build_candidate(
    base: dict[str, Any],
    forensic: dict[str, Any],
    g1: dict[str, Any],
    g2: dict[str, Any],
) -> dict[str, Any]:
    cid = str(base.get("case_id"))
    expected = [str(x) for x in (base.get("expected_documents") or [])]

    classification = forensic.get("classification", "UNKNOWN")
    dense_docs = docs_from_condition(forensic.get("dense", {}))
    ce_docs = docs_from_condition(forensic.get("crossencoder", {}))

    dense_expected_ranks = [
        i + 1 for i, doc in enumerate(dense_docs) if doc in expected
    ]
    ce_expected_ranks = [
        i + 1 for i, doc in enumerate(ce_docs) if doc in expected
    ]

    expected_set = set(expected)
    dense_set = set(dense_docs)
    ce_set = set(ce_docs)

    common = dense_set & ce_set
    ce_gained = ce_set - dense_set
    ce_lost = dense_set - ce_set

    rank_shift = None
    if dense_expected_ranks and ce_expected_ranks:
        rank_shift = min(dense_expected_ranks) - min(ce_expected_ranks)

    signals: list[str] = []
    score = 0

    # Multi-document dependency.
    if len(expected) >= 2:
        signals.append("MULTI_DOCUMENT_EXPECTATION")
        score += 4

    # Retrieval classification.
    if classification == "RANKING_IMPROVEMENT":
        signals.append("CE_RANKING_IMPROVEMENT")
        score += 8
    elif classification == "RANKING_REGRESSION":
        signals.append("CE_RANKING_REGRESSION")
        score += 8
    elif classification == "NEUTRAL_REORDER":
        signals.append("NEUTRAL_REORDER")
        score += 2

    # Candidate pool / evidence-set differences.
    if ce_gained:
        signals.append("CE_ADDS_DOCUMENTS")
        score += 4
    if ce_lost:
        signals.append("CE_REMOVES_DOCUMENTS")
        score += 3

    # Meaningful rank movement.
    if rank_shift is not None and rank_shift != 0:
        signals.append("EXPECTED_RANK_CHANGED")
        score += min(5, abs(rank_shift) * 2)

    # Expected evidence absent from final top-5.
    missing_dense = sorted(expected_set - dense_set)
    missing_ce = sorted(expected_set - ce_set)

    if missing_dense:
        signals.append("DENSE_MISSING_EXPECTED")
        score += 5
    if missing_ce:
        signals.append("CE_MISSING_EXPECTED")
        score += 5

    # G1 answer transitions, if available.
    g1_case = g1.get(cid, {})
    dense_correct = extract_g1_correct(g1_case, "dense")
    ce_correct = extract_g1_correct(g1_case, "dense_crossencoder")

    if dense_correct is not None and ce_correct is not None:
        if dense_correct != ce_correct:
            signals.append("ANSWER_CHANGED_G1")
            score += 12
        else:
            signals.append("ANSWER_UNCHANGED_G1")

    # G2 sensitivity.
    g2_case = g2.get(cid, {})
    expected_first = extract_g1_correct(g2_case, "expected_first")
    oracle = extract_g1_correct(g2_case, "oracle")

    if expected_first is not None and oracle is not None:
        if expected_first != oracle:
            signals.append("ORDERING_SENSITIVE_G2")
            score += 12

    if dense_correct is True and ce_correct is False:
        signals.append("CE_ANSWER_REGRESSION")
        score += 15

    return {
        "case_id": cid,
        "question": base.get("question"),
        "category": base.get("category"),
        "expected_documents": expected,
        "retrieval_classification": classification,
        "dense_documents": dense_docs,
        "crossencoder_documents": ce_docs,
        "ce_added_documents": sorted(ce_gained),
        "ce_removed_documents": sorted(ce_lost),
        "dense_expected_ranks": dense_expected_ranks,
        "crossencoder_expected_ranks": ce_expected_ranks,
        "expected_rank_shift": rank_shift,
        "dense_missing_expected": missing_dense,
        "crossencoder_missing_expected": missing_ce,
        "g1_dense_correct": dense_correct,
        "g1_crossencoder_correct": ce_correct,
        "g2_expected_first_correct": expected_first,
        "g2_oracle_correct": oracle,
        "signals": signals,
        "candidate_score": score,
    }


def main() -> None:
    required = [
        EVAL_CASES_PATH,
        FORENSIC_PATH,
        G1_PATH,
        G2_PATH,
    ]

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifact(s):\n" + "\n".join(missing)
        )

    dataset = load_json(EVAL_CASES_PATH)
    forensic_data = load_json(FORENSIC_PATH)
    g1_data = load_json(G1_PATH)
    g2_data = load_json(G2_PATH)

    dataset_cases = case_map(dataset)
    forensic_map = {
        str(case["case_id"]): case
        for case in forensic_cases(forensic_data)
        if case.get("case_id")
    }
    g1_map = get_g1_case_map(g1_data)
    g2_map = get_g1_case_map(g2_data)

    candidates = []

    for cid, base in dataset_cases.items():
        forensic = forensic_map.get(cid)
        if not forensic:
            continue

        candidates.append(
            build_candidate(
                base,
                forensic,
                g1_map,
                g2_map,
            )
        )

    candidates.sort(
        key=lambda x: (
            -x["candidate_score"],
            x["case_id"],
        )
    )

    classification_counts = Counter(
        x["retrieval_classification"] for x in candidates
    )

    signal_counts = Counter()
    for candidate in candidates:
        signal_counts.update(candidate["signals"])

    # Category recommendations are deliberately descriptive, not forced.
    # A case can belong to multiple diagnostic buckets.
    buckets = {
        "retrieval_improvement": [
            x for x in candidates
            if x["retrieval_classification"] == "RANKING_IMPROVEMENT"
        ],
        "retrieval_regression": [
            x for x in candidates
            if x["retrieval_classification"] == "RANKING_REGRESSION"
        ],
        "multi_evidence": [
            x for x in candidates
            if len(x["expected_documents"]) >= 2
        ],
        "ordering_sensitive": [
            x for x in candidates
            if "ORDERING_SENSITIVE_G2" in x["signals"]
        ],
        "answer_changed": [
            x for x in candidates
            if "ANSWER_CHANGED_G1" in x["signals"]
        ],
        "ce_answer_regression": [
            x for x in candidates
            if "CE_ANSWER_REGRESSION" in x["signals"]
        ],
        "dense_missing_expected": [
            x for x in candidates
            if "DENSE_MISSING_EXPECTED" in x["signals"]
        ],
    }

    output = {
        "experiment": "g3b_candidate_mining",
        "version": "0.1",
        "dataset_size": len(candidates),
        "api_calls_used": 0,
        "source_artifacts": {
            "evaluation_cases": str(EVAL_CASES_PATH),
            "retrieval_forensic": str(FORENSIC_PATH),
            "g1_judged": str(G1_PATH),
            "g2_judged": str(G2_PATH),
        },
        "classification_counts": dict(classification_counts),
        "signal_counts": dict(signal_counts),
        "bucket_counts": {
            key: len(value) for key, value in buckets.items()
        },
        "top_candidates": candidates[:30],
        "all_candidates": candidates,
        "method": {
            "purpose": (
                "Mine naturally difficult cases from the existing "
                "75-case benchmark before constructing synthetic cases."
            ),
            "no_llm_calls": True,
            "no_production_changes": True,
            "ranking_score_is_selection_heuristic": True,
            "candidate_score_is_not_a_quality_metric": True,
        },
        "next_step": (
            "Inspect top candidates and select a defensible G3B subset. "
            "Only after selection should retrieval-only screening be run."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 78)
    print("G3B — EXISTING-CASE CANDIDATE MINING")
    print("=" * 78)
    print(f"Cases analyzed: {len(candidates)}")
    print("API calls:      0")
    print()

    print("Retrieval classifications:")
    for key, value in sorted(classification_counts.items()):
        print(f"  {key:28s} {value}")

    print()
    print("Diagnostic buckets:")
    for key, value in buckets.items():
        print(f"  {key:28s} {len(value)}")

    print()
    print("Top 15 candidates:")
    for rank, candidate in enumerate(candidates[:15], start=1):
        signals = ", ".join(candidate["signals"][:4])
        print(
            f"  {rank:2d}. {candidate['case_id']:16s} "
            f"score={candidate['candidate_score']:2d} | {signals}"
        )

    print()
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 78)


if __name__ == "__main__":
    main()
