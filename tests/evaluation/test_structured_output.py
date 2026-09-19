from app.evaluation.structured_output import StructuredOutputValidator
from app.models.evaluation import AnswerJudgeResult


def _valid_output() -> dict:
    return {
        "answer_correct": True,
        "answer_grounded": True,
        "topics_covered": ["retrieval"],
        "topics_missing": [],
        "unsupported_claims": [],
        "reasoning": "The generated answer matches the evidence.",
    }


def test_valid_output_is_accepted():
    result = StructuredOutputValidator.validate(
        _valid_output(),
        AnswerJudgeResult,
    )

    assert result.valid is True
    assert result.value is not None
    assert result.errors == []


def test_missing_required_field_is_rejected():
    output = _valid_output()
    del output["reasoning"]

    result = StructuredOutputValidator.validate(
        output,
        AnswerJudgeResult,
    )

    assert result.valid is False
    assert result.value is None
    assert any(
        error["type"] == "missing"
        and error["loc"] == ("reasoning",)
        for error in result.errors
    )


def test_wrong_field_type_is_rejected():
    output = _valid_output()
    output["answer_correct"] = "true"

    result = StructuredOutputValidator.validate(
        output,
        AnswerJudgeResult,
    )

    assert result.valid is False
    assert result.value is None
    assert any(
        error["loc"] == ("answer_correct",)
        for error in result.errors
    )


def test_extra_field_is_rejected():
    output = _valid_output()
    output["unexpected"] = "value"

    result = StructuredOutputValidator.validate(
        output,
        AnswerJudgeResult,
    )

    assert result.valid is False
    assert result.value is None


def test_alias_keys_are_currently_rejected():
    output = {
        "correct": True,
        "grounded": True,
        "topics_covered": [],
        "topics_missing": [],
        "unsupported_claims": [],
        "reasoning": "The generated answer matches the evidence.",
    }

    result = StructuredOutputValidator.validate(
        output,
        AnswerJudgeResult,
    )

    assert result.valid is False
    assert result.value is None
