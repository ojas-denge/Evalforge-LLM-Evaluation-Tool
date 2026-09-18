import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.core.config import get_settings


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer_correct": {"type": "boolean"},
        "answer_grounded": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "answer_correct",
        "answer_grounded",
        "reasoning",
    ],
    "additionalProperties": False,
}


EXPERIMENTS = [
    {
        "name": "json_object",
        "response_format": {"type": "json_object"},
    },
    {
        "name": "json_object_repeat",
        "response_format": {"type": "json_object"},
    },
    {
        "name": "no_response_format",
        "response_format": None,
    },
    {
        "name": "no_response_format_repeat",
        "response_format": None,
    },
    {
        "name": "json_schema",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "evalforge_response",
                "schema": JUDGE_SCHEMA,
            },
        },
    },
]

OUTPUT_DIR = Path("logs") / "openrouter_controlled"


def main() -> None:
    settings = get_settings()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.llm_api_key}",
    }

    for run_number, experiment in enumerate(EXPERIMENTS, start=1):
        payload = {
            "model": settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Evaluate the statement provided by the user. "
                        "When asked for JSON, return a JSON object containing "
                        "answer_correct as a boolean, "
                        "answer_grounded as a boolean, "
                        "and reasoning as a string."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Evaluate this statement: "
                        "The sky is blue. "
                        "Return the evaluation."
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1000,
        }

        if experiment["response_format"] is not None:
            payload["response_format"] = experiment["response_format"]

        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "experiment": {
                "run_number": run_number,
                "name": experiment["name"],
                "timestamp_utc": timestamp,
            },
            "request": {
                "method": "POST",
                "url": f"{settings.llm_base_url}/chat/completions",
                "payload": payload,
            },
        }

        try:
            response = httpx.post(
                f"{settings.llm_base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=90.0,
            )

            latency_ms = (time.perf_counter() - started) * 1000

            record["response"] = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "raw_text": response.text,
                "latency_ms": latency_ms,
            }

            try:
                record["response"]["json"] = response.json()
            except ValueError:
                record["response"]["json"] = None

        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000

            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "latency_ms": latency_ms,
            }

        output_path = OUTPUT_DIR / f"{run_number:03d}_{experiment['name']}.json"

        output_path.write_text(
            json.dumps(
                record,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print("=" * 80)
        print(
            f"CONTROLLED OPERATION "
            f"{run_number}/{len(EXPERIMENTS)}"
        )
        print("=" * 80)
        print(f"Experiment: {experiment['name']}")

        if "response" not in record:
            error = record["error"]
            print("Status:    ERROR")
            print(f"Type:      {error['type']}")
            print(f"Error:     {error['message']}")
            print(f"Latency:   {error['latency_ms']:.2f} ms")
            print(f"Saved:     {output_path}")
            continue

        response_data = record["response"]
        data = response_data.get("json")

        print(f"Status:    {response_data['status_code']}")
        print(f"Latency:   {response_data['latency_ms']:.2f} ms")
        print(f"Saved:     {output_path}")

        if isinstance(data, dict):
            print(f"Model:     {data.get('model')}")
            print(f"Provider:  {data.get('provider')}")

            choices = data.get("choices") or []

            if choices:
                choice = choices[0]
                print(f"Finish:    {choice.get('finish_reason')}")

                message = choice.get("message") or {}
                content = message.get("content")

                print(
                    "Content:   "
                    f"{repr(content)}"
                )

                if isinstance(content, str):
                    try:
                        parsed_content = json.loads(content)
                        print("JSON:      VALID")
                        print(
                            "JSON keys: "
                            f"{list(parsed_content.keys())}"
                            if isinstance(parsed_content, dict)
                            else "JSON type: "
                            f"{type(parsed_content).__name__}"
                        )
                    except json.JSONDecodeError:
                        print("JSON:      INVALID")

                print(
                    "Reasoning: "
                    f"{'present' if message.get('reasoning') else 'absent'}"
                )

            usage = data.get("usage") or {}
            print(
                "Tokens:    "
                f"{usage.get('total_tokens', 0)}"
            )

        else:
            print("Response JSON: unavailable")

        if run_number < len(EXPERIMENTS):
            print("\nWaiting before next operation...\n")
            time.sleep(1)


if __name__ == "__main__":
    main()
