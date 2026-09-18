import pytest
from app.evaluation.perturbation import (
    PerturbationResult,
    PerturbationType,
    compute_perturbation_stability,
    reorder_evidence,
)
from app.models.evaluation import RetrievedEvidence

def test_reorder_evidence_reverses_order():
    evidence = [
        RetrievedEvidence(rank=1, chunk_id="c1", document_id="d1", distance=0.1, text="first"),
        RetrievedEvidence(rank=2, chunk_id="c2", document_id="d2", distance=0.2, text="second"),
        RetrievedEvidence(rank=3, chunk_id="c3", document_id="d3", distance=0.3, text="third"),
    ]
    reordered = reorder_evidence(evidence)
    assert len(reordered) == 3
    assert reordered[0].chunk_id == "c3"
    assert reordered[0].rank == 1
    assert reordered[1].chunk_id == "c2"
    assert reordered[1].rank == 2
    assert reordered[2].chunk_id == "c1"
    assert reordered[2].rank == 3

def test_reorder_evidence_empty_list():
    assert reorder_evidence([]) == []

def test_reorder_evidence_single_item():
    evidence = [
        RetrievedEvidence(rank=1, chunk_id="c1", document_id="d1", distance=0.1, text="only"),
    ]
    reordered = reorder_evidence(evidence)
    assert len(reordered) == 1
    assert reordered[0].chunk_id == "c1"

def test_compute_perturbation_stability_all_stable():
    results = [
        PerturbationResult(
            case_id="c1",
            perturbation_type=PerturbationType.EVIDENCE_REORDER,
            original_correct=True,
            original_grounded=True,
            perturbed_correct=True,
            perturbed_grounded=True,
            is_stable=True,
        ),
        PerturbationResult(
            case_id="c2",
            perturbation_type=PerturbationType.EVIDENCE_REORDER,
            original_correct=False,
            original_grounded=False,
            perturbed_correct=False,
            perturbed_grounded=False,
            is_stable=True,
        ),
    ]
    assert compute_perturbation_stability(results) == 1.0

def test_compute_perturbation_stability_partial():
    results = [
        PerturbationResult(
            case_id="c1",
            perturbation_type=PerturbationType.EVIDENCE_REORDER,
            original_correct=True,
            original_grounded=True,
            perturbed_correct=True,
            perturbed_grounded=True,
            is_stable=True,
        ),
        PerturbationResult(
            case_id="c2",
            perturbation_type=PerturbationType.EVIDENCE_REORDER,
            original_correct=True,
            original_grounded=True,
            perturbed_correct=False,
            perturbed_grounded=True,
            is_stable=False,
        ),
    ]
    assert compute_perturbation_stability(results) == 0.5

def test_compute_perturbation_stability_rejects_empty():
    with pytest.raises(ValueError, match="No perturbation results"):
        compute_perturbation_stability([])
