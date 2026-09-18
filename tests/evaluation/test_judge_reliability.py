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
