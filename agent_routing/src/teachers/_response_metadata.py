"""Provider response evidence without response bodies or credentials."""
from __future__ import annotations

from collections.abc import Mapping


def _get(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _usage_value(value):
    """Keep numeric usage details, including cache/reasoning token breakdowns."""
    if value is None or type(value) in (int, float, bool):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif not isinstance(value, Mapping) and hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, Mapping):
        return {str(key): _usage_value(item) for key, item in value.items()
                if not str(key).startswith("_")}
    return None


def response_metadata(response, *, finish_reason, latency_seconds, request_attempts,
                      input_field="prompt_tokens", output_field="completion_tokens"):
    provider_usage = _get(response, "usage")
    def count(name):
        value = _get(provider_usage, name)
        return value if type(value) is int and value >= 0 else None
    prompt, completion, total = count(input_field), count(output_field), count("total_tokens")
    # Anthropic has no total_tokens field. Its input_tokens excludes cache reads
    # and writes, so do not report input+output as a complete billed-token total.
    return {"id": _get(response, "id"), "model": _get(response, "model"),
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total},
            "provider_usage": _usage_value(provider_usage), "finish_reason": finish_reason,
            "latency_seconds": float(latency_seconds), "request_attempts": request_attempts,
            "system_fingerprint": _get(response, "system_fingerprint")}


def failure_message(provider, error, request_attempts):
    # SDK errors may include headers, echoed prompts, or credentials. Preserve
    # only the exception class and numeric status in the externally logged text.
    status = getattr(error, "status_code", None)
    suffix = f", HTTP {status}" if type(status) is int else ""
    return f"{provider} chat failed after {request_attempts} request(s): {type(error).__name__}{suffix}"
