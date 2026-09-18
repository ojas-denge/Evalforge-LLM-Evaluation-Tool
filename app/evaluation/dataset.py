import json
from pathlib import Path

from app.models.evaluation import EvaluationCase


class EvaluationDataset:
    def __init__(self, cases: list[EvaluationCase]) -> None:
        self.cases = cases

    @classmethod
    def load(cls, path: str = "data/evaluation_cases.json"):
        dataset_path = Path(path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Evaluation dataset not found: {dataset_path}"
            )

        raw_data = json.loads(
            dataset_path.read_text(encoding="utf-8")
        )

        if not isinstance(raw_data, list):
            raise ValueError(
                "Evaluation dataset must contain a JSON list"
            )

        cases = [
            EvaluationCase.model_validate(item)
            for item in raw_data
        ]

        cls._validate_unique_ids(cases)

        return cls(cases)

    @staticmethod
    def _validate_unique_ids(
        cases: list[EvaluationCase],
    ) -> None:
        case_ids = [case.case_id for case in cases]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "Evaluation case IDs must be unique"
            )

    def __len__(self) -> int:
        return len(self.cases)
