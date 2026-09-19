from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class StructuredValidationResult:
    """Result of validating structured model output."""

    valid: bool
    value: BaseModel | None
    errors: list[dict[str, Any]]


class StructuredOutputValidator:
    """Strictly validate structured output against a Pydantic model."""

    @staticmethod
    def validate(
        output: Any,
        model: type[ModelT],
    ) -> StructuredValidationResult:
        if not isinstance(output, dict):
            return StructuredValidationResult(
                valid=False,
                value=None,
                errors=[
                    {
                        "type": "model_type",
                        "loc": (),
                        "msg": "Structured output must be a JSON object",
                        "input": output,
                    }
                ],
            )

        allowed_fields = set(model.model_fields)
        unexpected_fields = sorted(
            set(output) - allowed_fields
        )

        if unexpected_fields:
            errors = [
                {
                    "type": "extra_forbidden",
                    "loc": (field,),
                    "msg": "Extra inputs are not permitted",
                    "input": output[field],
                }
                for field in unexpected_fields
            ]

            return StructuredValidationResult(
                valid=False,
                value=None,
                errors=errors,
            )

        try:
            validated = model.model_validate(output)

            return StructuredValidationResult(
                valid=True,
                value=validated,
                errors=[],
            )

        except ValidationError as exc:
            return StructuredValidationResult(
                valid=False,
                value=None,
                errors=exc.errors(),
            )
