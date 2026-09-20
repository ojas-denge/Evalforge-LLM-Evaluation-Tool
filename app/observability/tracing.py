from contextlib import contextmanager
from typing import Any, Iterator

from app.core.config import get_settings


def classify_error(exc: Exception) -> dict[str, Any]:
    """Return safe, non-secret telemetry fields for an exception."""
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)

    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    exception_name = type(exc).__name__
    message = str(exc).lower()

    if status_code == 429 or any(
        marker in message
        for marker in (
            "429",
            "rate limit",
            "rate_limit",
            "too many requests",
            "quota",
            "resource exhausted",
        )
    ):
        category = "rate_limit"
        retryable = True
    elif (
        "timeout" in message
        or exception_name in {"TimeoutException", "ReadTimeout", "ConnectTimeout"}
    ):
        category = "timeout"
        retryable = True
    elif status_code is not None:
        category = "provider_http_error"
        retryable = status_code >= 500
    elif isinstance(exc, (ValueError, TypeError)):
        category = "validation"
        retryable = False
    else:
        category = "internal_error"
        retryable = False

    return {
        "status": "error",
        "error_type": exception_name,
        "error_category": category,
        "http_status": status_code,
        "retryable": retryable,
    }


class Tracer:
    """Optional observability layer for EvalForge."""

    def __init__(self) -> None:
        settings = get_settings()

        self.enabled = settings.langfuse_enabled
        self.capture_content = settings.langfuse_capture_content
        self.client = None

        if self.enabled:
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )

    def redact(self, value: Any) -> Any:
        """Apply the configured telemetry content policy."""
        if self.capture_content:
            return value
        return "[redacted]"

    @contextmanager
    def _observation(
        self,
        *,
        name: str,
        as_type: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        if not self.enabled:
            yield None
            return

        with self.client.start_as_current_observation(
            name=name,
            as_type=as_type,
            input=input,
            metadata=metadata,
        ) as observation:
            try:
                yield observation
            except Exception as exc:
                observation.update(
                    output=classify_error(exc),
                    metadata=classify_error(exc),
                )
                raise

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        with self._observation(
            name=name,
            as_type="chain",
            input=input,
            metadata=metadata,
        ) as observation:
            yield observation

    @contextmanager
    def retrieval(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        with self._observation(
            name=name,
            as_type="retriever",
            input=input,
            metadata=metadata,
        ) as observation:
            yield observation

    @contextmanager
    def generation(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        with self._observation(
            name=name,
            as_type="generation",
            input=input,
            metadata=metadata,
        ) as observation:
            yield observation

    @contextmanager
    def judge(
        self,
        name: str,
        *,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Any]:
        with self._observation(
            name=name,
            as_type="chain",
            input=input,
            metadata=metadata,
        ) as observation:
            yield observation

    def flush(self) -> None:
        if self.client is not None:
            self.client.flush()
