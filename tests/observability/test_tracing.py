from app.observability.tracing import classify_error


def test_classify_error_marks_rate_limit_retryable():
    class RateLimitError(Exception):
        status_code = 429

    result = classify_error(RateLimitError())

    assert result == {
        "status": "error",
        "error_type": "RateLimitError",
        "error_category": "rate_limit",
        "http_status": 429,
        "retryable": True,
    }


def test_classify_error_does_not_include_exception_message():
    class ProviderError(Exception):
        status_code = 500

        def __str__(self):
            return "secret-looking provider payload"

    result = classify_error(ProviderError())

    assert "secret-looking provider payload" not in result.values()
    assert result["error_category"] == "provider_http_error"
    assert result["retryable"] is True
