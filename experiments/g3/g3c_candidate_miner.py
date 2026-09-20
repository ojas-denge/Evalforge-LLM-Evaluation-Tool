from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

DATASET = Path("data/evaluation_cases.json")
FORENSIC = Path("logs/dense_vs_crossencoder_forensic.json")
OUTPUT = Path("logs/g3c_candidate_mining.json")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ranks_for_expected(documents: list[str], expected: list[str]) -> list[int]:
    expected_set = set(expected)
    return [
        rank
        for rank, doc in enumerate(documents, start=1)
        if doc in expected_set
    ]


def main() -> None:
    cases = load_json(DATASET)
    forensic = load_json(FORENSIC)

    forensic_cases = forensic.get("cases", forensic)
    forensic_by_id = {
        item["case_id"]: item
        for item in forensic_cases
        if "case_id" in item
    }

    candidates = []

    for case in cases:
        cid = case["case_id"]
        f = forensic_by_id.get(cid)
        if not f:
            continue

        expected = case.get("expected_documents", [])
        dense_docs = f.get("dense", {}).get("documents", [])
        ce_docs = f.get("crossencoder", {}).get("documents", [])

        if not dense_docs or not ce_docs:
            continue

        dense_ranks = ranks_for_expected(dense_docs, expected)
        ce_ranks = ranks_for_expected(ce_docs, expected)

        dense_set = set(dense_docs)
        ce_set = set(ce_docs)

        signals = []

        # G3C wants cases where the answer should depend on evidence,
        # rather than merely on a single obvious document.
        if len(expected) >= 2:
            signals.append("MULTI_DOCUMENT")

        if dense_set != ce_set:
            signals.append("EVIDENCE_SET_CHANGE")

        if dense_ranks != ce_ranks:
            signals.append("EXPECTED_RANK_CHANGE")

        if dense_ranks and ce_ranks:
            dense_first = min(dense_ranks)
            ce_first = min(ce_ranks)
            if dense_first != ce_first:
                signals.append("FIRST_RELEVANT_RANK_CHANGE")

        # Cases with multiple expected documents plus rank movement are
        # especially useful for causal propagation testing.
        score = 0
        score += 5 if len(expected) >= 2 else 0
        score += 5 if "EVIDENCE_SET_CHANGE" in signals else 0
        score += 4 if "FIRST_RELEVANT_RANK_CHANGE" in signals else 0
        score += 2 if "EXPECTED_RANK_CHANGE" in signals else 0

        if score == 0:
            continue

        candidates.append({
            "case_id": cid,
            "question": case["question"],
            "category": case["category"],
            "difficulty": case["difficulty"],
            "expected_documents": expected,
            "dense_documents": dense_docs,
            "crossencoder_documents": ce_docs,
            "dense_expected_ranks": dense_ranks,
            "crossencoder_expected_ranks": ce_ranks,
            "signals": signals,
            "score": score,
            "g3c_review_required": True,
            "review_fields": {
                "evidence_dependency": None,
                "single_document_sufficient": None,
                "distractor_is_plausible": None,
                "answer_inferable_without_retrieval": None,
                "critical_relationship": None,
                "why_dense_and_ce_should_differ": None,
            },
        })

    candidates.sort(
        key=lambda x: (-x["score"], x["case_id"])
    )

    output = {
        "experiment": "g3c_candidate_mining",
        "version": "0.1",
        "api_calls_used": 0,
        "purpose": (
            "Identify existing benchmark cases that are plausible "
            "candidates for a retrieval-dependent G3C diagnostic set."
        ),
        "selection_rules": {
            "prefer_multi_document": True,
            "prefer_evidence_set_change": True,
            "prefer_expected_rank_change": True,
            "manual_review_required": True,
            "no_llm_generation_during_mining": True,
        },
        "summary": {
            "dataset_size": len(cases),
            "candidate_count": len(candidates),
            "signal_counts": dict(
                Counter(
                    signal
                    for candidate in candidates
                    for signal in candidate["signals"]
                )
            ),
        },
        "candidates": candidates,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("G3C candidate mining complete")
    print(f"Dataset cases : {len(cases)}")
    print(f"Candidates    : {len(candidates)}")
    print(f"Output        : {OUTPUT}")


if __name__ == "__main__":
    main()
