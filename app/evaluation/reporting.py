def print_comparison(comparison):
    improved = []
    degraded = []
    failure_transitions = []

    for change in comparison.case_changes:
        baseline_failure = change.baseline_failure_type
        candidate_failure = change.candidate_failure_type

        if baseline_failure != "PASS" and candidate_failure == "PASS":
            improved.append(change)

        elif baseline_failure == "PASS" and candidate_failure != "PASS":
            degraded.append(change)

        elif (
            baseline_failure != "PASS"
            and candidate_failure != "PASS"
            and baseline_failure != candidate_failure
        ):
            failure_transitions.append(change)

        elif (
            baseline_failure == candidate_failure == "PASS"
            and (
                change.hit_at_1_delta != 0
                or change.mrr_delta != 0
            )
        ):
            if change.hit_at_1_delta > 0 or change.mrr_delta > 0:
                improved.append(change)
            elif change.hit_at_1_delta < 0 or change.mrr_delta < 0:
                degraded.append(change)

    print("\n=== Aggregate Metric Deltas ===")

    for metric, delta in comparison.metric_deltas.items():
        print(f"{metric}: {delta:+.4f}")

    print("\n=== Improved ===")

    if not improved:
        print("None")

    for change in improved:
        print(f"\n{change.case_id}")
        print(f"  Hit@1: {change.hit_at_1_delta:+.2f}")
        print(f"  MRR: {change.mrr_delta:+.2f}")
        print(
            f"  Failure: "
            f"{change.baseline_failure_type}"
            f" -> "
            f"{change.candidate_failure_type}"
        )
        print(
            f"  First relevant rank: "
            f"{change.baseline_first_relevant_rank}"
            f" -> "
            f"{change.candidate_first_relevant_rank}"
        )

    print("\n=== Degraded ===")

    if not degraded:
        print("None")

    for change in degraded:
        print(f"\n{change.case_id}")
        print(f"  Hit@1: {change.hit_at_1_delta:+.2f}")
        print(f"  MRR: {change.mrr_delta:+.2f}")
        print(
            f"  Failure: "
            f"{change.baseline_failure_type}"
            f" -> "
            f"{change.candidate_failure_type}"
        )
        print(
            f"  First relevant rank: "
            f"{change.baseline_first_relevant_rank}"
            f" -> "
            f"{change.candidate_first_relevant_rank}"
        )
        print(
            f"  Missing documents: "
            f"{change.baseline_missing_documents}"
            f" -> "
            f"{change.candidate_missing_documents}"
        )

    print("\n=== Failure Transitions ===")

    if not failure_transitions:
        print("None")

    for change in failure_transitions:
        print(f"\n{change.case_id}")
        print(
            f"  Failure: "
            f"{change.baseline_failure_type}"
            f" -> "
            f"{change.candidate_failure_type}"
        )
        print(f"  MRR: {change.mrr_delta:+.2f}")
        print(
            f"  First relevant rank: "
            f"{change.baseline_first_relevant_rank}"
            f" -> "
            f"{change.candidate_first_relevant_rank}"
        )
        print(
            f"  Missing documents: "
            f"{change.baseline_missing_documents}"
            f" -> "
            f"{change.candidate_missing_documents}"
        )
