def test_reciprocal_rank_at_rank_1():
    from app.evaluation.metrics import reciprocal_rank

    assert reciprocal_rank(
        ["doc-a"],
        ["doc-a", "doc-b", "doc-c"],
    ) == 1.0


def test_reciprocal_rank_at_rank_2():
    from app.evaluation.metrics import reciprocal_rank

    assert reciprocal_rank(
        ["doc-a"],
        ["doc-b", "doc-a", "doc-c"],
    ) == 0.5


def test_reciprocal_rank_at_rank_3():
    from app.evaluation.metrics import reciprocal_rank

    assert reciprocal_rank(
        ["doc-a"],
        ["doc-b", "doc-c", "doc-a"],
    ) == 1 / 3


def test_reciprocal_rank_when_not_retrieved():
    from app.evaluation.metrics import reciprocal_rank

    assert reciprocal_rank(
        ["doc-a"],
        ["doc-b", "doc-c"],
    ) == 0.0


def test_reciprocal_rank_uses_first_relevant_document():
    from app.evaluation.metrics import reciprocal_rank

    assert reciprocal_rank(
        ["doc-a", "doc-c"],
        ["doc-b", "doc-c", "doc-a"],
    ) == 0.5
