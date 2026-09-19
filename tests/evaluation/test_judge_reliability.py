import pytest
from app.evaluation.judge_reliability import (
    compute_repeatability,
    compute_inter_judge_agreement,
    compute_gold_set_agreement,
)

def test_compute_repeatability_perfect_agreement():
    run_1 = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": False, "answer_grounded": True},
    ]
    run_2 = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": False, "answer_grounded": True},
    ]
    assert compute_repeatability([run_1, run_2]) == 1.0

def test_compute_repeatability_partial_agreement():
    run_1 = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": True, "answer_grounded": True},
    ]
    run_2 = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": False, "answer_grounded": True},
    ]
    assert compute_repeatability([run_1, run_2]) == 0.5

def test_compute_repeatability_rejects_single_run():
    with pytest.raises(ValueError, match="at least 2 runs"):
        compute_repeatability([[{"case_id": "c1", "answer_correct": True, "answer_grounded": True}]])

def test_compute_repeatability_rejects_empty():
    with pytest.raises(ValueError, match="at least 2 runs"):
        compute_repeatability([])

def test_compute_repeatability_three_runs():
    run_1 = [{"case_id": "c1", "answer_correct": True, "answer_grounded": True}]
    run_2 = [{"case_id": "c1", "answer_correct": True, "answer_grounded": True}]
    run_3 = [{"case_id": "c1", "answer_correct": False, "answer_grounded": True}]
    assert compute_repeatability([run_1, run_2, run_3]) == 0.0

def test_compute_gold_set_agreement_perfect():
    judge = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": False, "answer_grounded": False},
    ]
    gold = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": False, "answer_grounded": False},
    ]
    assert compute_gold_set_agreement(judge, gold) == 1.0

def test_compute_gold_set_agreement_partial():
    judge = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": True, "answer_grounded": True},
    ]
    gold = [
        {"case_id": "c1", "answer_correct": True, "answer_grounded": True},
        {"case_id": "c2", "answer_correct": False, "answer_grounded": True},
    ]
    assert compute_gold_set_agreement(judge, gold) == 0.5

def test_compute_gold_set_agreement_rejects_no_overlap():
    judge = [{"case_id": "c1", "answer_correct": True, "answer_grounded": True}]
    gold = [{"case_id": "c2", "answer_correct": True, "answer_grounded": True}]
    with pytest.raises(ValueError, match="No common case IDs"):
        compute_gold_set_agreement(judge, gold)
from app.evaluation.judge_reliability import compute_judge_reliability


def test_compute_judge_reliability_tracks_final_validity_recovery_and_failures():
    results = [
        {
            "case_id": "c1",
            "judge_verdict": {
                "status": "valid",
                "structured_output_valid": True,
                "judge_metadata": {
                    "judge_recovered": False,
                },
            },
        },
        {
            "case_id": "c2",
            "judge_verdict": {
                "status": "valid",
                "structured_output_valid": True,
                "judge_metadata": {
                    "judge_recovered": True,
                    "initial_failure_type": "schema_invalid",
                },
            },
        },
        {
            "case_id": "c3",
            "judge_verdict": {
                "status": "system_error",
                "structured_output_valid": False,
                "judge_metadata": {
                    "judge_recovered": False,
                    "initial_failure_type": "missing_content",
                },
            },
        },
        {
            "case_id": "c4",
            "judge_verdict": None,
        },
    ]

    report = compute_judge_reliability(results)

    assert report.structured_output_validity_rate == 2 / 3
    assert report.recovery_rate == 0.5
    assert report.failure_rate == 1 / 3
    assert report.total_cases == 3
    assert report.failure_count == 1


def test_compute_judge_reliability_with_no_evaluated_cases():
    results = [
        {"case_id": "c1", "judge_verdict": None},
        {"case_id": "c2", "judge_verdict": None},
    ]

    report = compute_judge_reliability(results)

    assert report.structured_output_validity_rate == 0.0
    assert report.recovery_rate == 0.0
    assert report.failure_rate == 0.0
    assert report.total_cases == 0
    assert report.failure_count == 0


def test_compute_judge_reliability_calculates_recovery_from_initial_failures():
    results = [
        {
            "case_id": "c1",
            "judge_verdict": {
                "status": "valid",
                "structured_output_valid": True,
                "judge_metadata": {
                    "judge_recovered": True,
                    "initial_failure_type": "schema_invalid",
                },
            },
        },
        {
            "case_id": "c2",
            "judge_verdict": {
                "status": "valid",
                "structured_output_valid": True,
                "judge_metadata": {
                    "judge_recovered": True,
                    "initial_failure_type": "parse_failed",
                },
            },
        },
        {
            "case_id": "c3",
            "judge_verdict": {
                "status": "system_error",
                "structured_output_valid": False,
                "judge_metadata": {
                    "judge_recovered": False,
                    "initial_failure_type": "missing_content",
                },
            },
        },
        {
            "case_id": "c4",
            "judge_verdict": {
                "status": "valid",
                "structured_output_valid": True,
                "judge_metadata": {
                    "judge_recovered": False,
                },
            },
        },
    ]

    report = compute_judge_reliability(results)

    assert report.structured_output_validity_rate == 0.75
    assert report.recovery_rate == 2 / 3
    assert report.failure_rate == 0.25
    assert report.total_cases == 4
    assert report.failure_count == 1

def test_compute_judge_reliability_does_not_count_invalid_recovery():
    results = [
        {
            "case_id": "c1",
            "judge_verdict": {
                "status": "system_error",
                "structured_output_valid": False,
                "judge_metadata": {
                    "judge_recovered": True,
                    "initial_failure_type": "schema_invalid",
                },
            },
        },
    ]

    report = compute_judge_reliability(results)

    assert report.structured_output_validity_rate == 0.0
    assert report.recovery_rate == 0.0
    assert report.failure_rate == 1.0
    assert report.failure_count == 1
