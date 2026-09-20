import json
from collections import Counter


def load_json(path: str) -> dict:
    with open(path, "rb") as file:
        raw = file.read()

    if raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8")

    return json.loads(text)


def main() -> None:
    data = load_json("candidate_diagnosis.json")
    cases = data["cases"]

    counts = Counter()
    candidate_gain_examples = []
    reranking_effect_examples = []

    for case in cases:
        k5 = case["k5"]
        k10 = case["k10"]

        k5_hits = set(k5["candidate_hits"])
        k10_hits = set(k10["candidate_hits"])

        gained = k10_hits - k5_hits

        if gained:
            counts["K10_ADDS_EXPECTED_EVIDENCE"] += 1

            if len(candidate_gain_examples) < 10:
                candidate_gain_examples.append(
                    {
                        "case_id": case["case_id"],
                        "gained": sorted(gained),
                        "k5_final_ranks": k5["final_ranks"],
                        "k10_final_ranks": k10["final_ranks"],
                    }
                )

        elif k5_hits == k10_hits:
            counts["NO_CANDIDATE_GAIN"] += 1

            if (
                k5["final_documents"] != k10["final_documents"]
                and len(reranking_effect_examples) < 10
            ):
                reranking_effect_examples.append(
                    {
                        "case_id": case["case_id"],
                        "expected": sorted(
                            k5["expected_documents"]
                        ),
                        "k5_final_ranks": k5["final_ranks"],
                        "k10_final_ranks": k10["final_ranks"],
                    }
                )

        else:
            counts["OTHER_CANDIDATE_DIFFERENCE"] += 1

    output = {
        "total_cases": data["total_cases"],
        "different_final_lists": data["different_cases"],
        "summary": dict(counts),
        "candidate_gain_examples": candidate_gain_examples,
        "same_candidate_pool_examples": reranking_effect_examples,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
