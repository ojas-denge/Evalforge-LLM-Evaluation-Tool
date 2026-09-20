from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Any

FORENSIC_PATH = Path("logs/dense_vs_crossencoder_forensic.json")
G1_JUDGED_PATH = Path("logs/g1_judged.json")
G2_JUDGED_PATH = Path("logs/g2_judged.json")
OUTPUT_PATH = Path("logs/g3a_failure_mining.json")

def load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def cases_for(data: dict[str, Any], condition: str) -> dict[str, Any]:
    return data.get("results", {}).get(condition, {})

def state(x: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_correct": x.get("answer_correct"),
        "answer_grounded": x.get("answer_grounded"),
        "topics_missing": x.get("topics_missing", []),
        "unsupported_claims": x.get("unsupported_claims", []),
        "reasoning": x.get("reasoning"),
    }

def transition(a, b):
    if a is True and b is True: return "CORRECT -> CORRECT"
    if a is True and b is False: return "CORRECT -> INCORRECT"
    if a is False and b is True: return "INCORRECT -> CORRECT"
    if a is False and b is False: return "INCORRECT -> INCORRECT"
    return "UNKNOWN"

def sensitivity(dense, oracle):
    if dense is False and oracle is True: return "RETRIEVAL_SENSITIVE"
    if dense is True and oracle is True: return "RETRIEVAL_INSENSITIVE"
    if dense is False and oracle is False: return "GENERATION_LIMITED"
    if dense is True and oracle is False: return "ORACLE_PARADOX"
    return "UNKNOWN"

def main():
    forensic = load(FORENSIC_PATH)
    g1 = load(G1_JUDGED_PATH)
    g2 = load(G2_JUDGED_PATH)

    fcases = {x["case_id"]: x for x in forensic.get("cases", [])}
    gd = cases_for(g1, "dense")
    gc = cases_for(g1, "dense_crossencoder")
    d = cases_for(g2, "dense")
    c = cases_for(g2, "dense_crossencoder")
    e = cases_for(g2, "expected_first")
    o = cases_for(g2, "oracle")

    ids = sorted(set(fcases) | set(gd) | set(gc) | set(d) | set(c) | set(e) | set(o))
    cases = []

    for cid in ids:
        f = fcases.get(cid, {})
        dv = d.get(cid, gd.get(cid, {}))
        cv = c.get(cid, gc.get(cid, {}))
        ev = e.get(cid, {})
        ov = o.get(cid, {})
        dc = dv.get("answer_correct")
        cc = cv.get("answer_correct")
        ec = ev.get("answer_correct")
        oc = ov.get("answer_correct")

        cases.append({
            "case_id": cid,
            "question": f.get("question"),
            "expected_documents": f.get("expected_documents", []),
            "retrieval_classification": f.get("classification"),
            "dense_failure_type": f.get("dense", {}).get("failure_type"),
            "crossencoder_failure_type": f.get("crossencoder", {}).get("failure_type"),
            "dense_documents": f.get("dense", {}).get("documents", []),
            "crossencoder_documents": f.get("crossencoder", {}).get("documents", []),
            "answer_correctness": {
                "dense": dc, "dense_crossencoder": cc,
                "expected_first": ec, "oracle": oc
            },
            "answer_states": {
                "dense": state(dv), "dense_crossencoder": state(cv),
                "expected_first": state(ev), "oracle": state(ov)
            },
            "transitions": {
                "dense_to_crossencoder": transition(dc, cc),
                "dense_to_expected_first": transition(dc, ec),
                "dense_to_oracle": transition(dc, oc),
                "expected_first_to_oracle": transition(ec, oc),
            },
            "sensitivity_class": sensitivity(dc, oc),
        })

    conditions = ("dense", "dense_crossencoder", "expected_first", "oracle")
    correctness = {
        k: {
            "correct": sum(x["answer_correctness"][k] is True for x in cases),
            "incorrect": sum(x["answer_correctness"][k] is False for x in cases),
            "unknown": sum(x["answer_correctness"][k] is None for x in cases),
        } for k in conditions
    }

    transition_counts = {
        k: dict(Counter(x["transitions"][k] for x in cases))
        for k in (
            "dense_to_crossencoder", "dense_to_expected_first",
            "dense_to_oracle", "expected_first_to_oracle"
        )
    }

    output = {
        "experiment": "g3a_failure_mining",
        "dataset_size": len(cases),
        "sources": {
            "retrieval_forensic": str(FORENSIC_PATH),
            "g1_judged": str(G1_JUDGED_PATH),
            "g2_judged": str(G2_JUDGED_PATH),
        },
        "aggregate": {
            "answer_correctness": correctness,
            "retrieval_classification": dict(Counter(
                x["retrieval_classification"] for x in cases
                if x["retrieval_classification"] is not None
            )),
            "sensitivity_classification": dict(Counter(
                x["sensitivity_class"] for x in cases
            )),
            "transitions": transition_counts,
        },
        "retrieval_sensitive_cases": [
            x for x in cases if x["sensitivity_class"] == "RETRIEVAL_SENSITIVE"
        ],
        "oracle_paradox_cases": [
            x for x in cases if x["sensitivity_class"] == "ORACLE_PARADOX"
        ],
        "generation_limited_cases": [
            x for x in cases if x["sensitivity_class"] == "GENERATION_LIMITED"
        ],
        "cases": cases,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("G3A FAILURE MINING")
    print(f"Cases: {len(cases)}")
    print("\nANSWER CORRECTNESS")
    for k, v in correctness.items():
        print(f"{k:20s} correct={v['correct']:2d} incorrect={v['incorrect']:2d}")
    print("\nSENSITIVITY")
    print(dict(Counter(x["sensitivity_class"] for x in cases)))
    print("\nDENSE -> ORACLE")
    print(transition_counts["dense_to_oracle"])
    print(f"\nOutput: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
