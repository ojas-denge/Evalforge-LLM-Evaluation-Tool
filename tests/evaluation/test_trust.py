from app.evaluation.trust import TrustTier, ConflictType, TrustAssessment


def test_trust_tier_enum():
    assert TrustTier.DETERMINISTIC_FACT == "tier_1"


def test_conflict_type_enum():
    assert ConflictType.JUDGE_CONTRADICTS_RETRIEVAL == "judge_contradicts_retrieval"


def test_trust_assessment_model():
    assessment = TrustAssessment(
        final_verdict="PASS",
        confidence="HIGH"
    )
    assert assessment.final_verdict == "PASS"
    assert assessment.confidence == "HIGH"
    assert assessment.conflicts == []
    assert assessment.tier_1_signals == {}
