"""Offline provider doubles: no credentials, imports, or network are required."""
import json
import sys
import traceback
from types import SimpleNamespace

import pytest

from src.teachers.base import build_teacher_client


PROVIDERS = ("openai", "deepseek", "anthropic")


def fake_sdk(monkeypatch, provider, outcomes):
    calls, constructors, sleeps = [], [], []
    pending = iter(outcomes)
    def create(**kwargs):
        calls.append(kwargs)
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    def constructor(**kwargs):
        constructors.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                               messages=SimpleNamespace(create=create))
    module = "anthropic" if provider == "anthropic" else "openai"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(OpenAI=constructor, Anthropic=constructor))
    monkeypatch.setattr(f"src.teachers.{provider}_client.time.sleep", sleeps.append)
    return calls, constructors, sleeps


def response(provider, *, metadata=True):
    if provider == "anthropic":
        value = SimpleNamespace(content=[SimpleNamespace(type="text", text="  first "),
                                          SimpleNamespace(type="thinking", thinking="not output"),
                                          SimpleNamespace(type="text", text="second  ")])
        if metadata:
            value.stop_reason = "max_tokens"
            value.usage = SimpleNamespace(input_tokens=12, output_tokens=7,
                                          cache_read_input_tokens=3, cache_creation_input_tokens=2)
    else:
        choice = SimpleNamespace(message=SimpleNamespace(content="  first second  "))
        value = SimpleNamespace(choices=[choice])
        if metadata:
            choice.finish_reason = "length"
            value.usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=2))
    if metadata:
        value.id, value.model, value.system_fingerprint = "request-provider-id", "actual-model-version", "fp"
    return value


@pytest.mark.parametrize("provider", PROVIDERS)
def test_teacher_preserves_text_and_actual_provider_evidence(monkeypatch, provider):
    # Load the wrapper before replacing its provider SDK module.
    __import__(f"src.teachers.{provider}_client")
    calls, constructors, sleeps = fake_sdk(monkeypatch, provider, [response(provider)])
    client = build_teacher_client(provider, "requested-model", api_key="offline-test-placeholder", max_retries=0)
    result = client.chat([{"role": "system", "content": "system one"},
                          {"role": "system", "content": "system two"},
                          {"role": "user", "content": "question"}], max_tokens=23)
    assert result.text == "first second"
    assert result.model == "requested-model" and result.provider == provider
    assert constructors[0]["max_retries"] == 0 and len(calls) == 1 and sleeps == []
    assert calls[0]["max_tokens"] == 23
    raw = result.raw
    assert raw["model"] == "actual-model-version" and raw["id"] == "request-provider-id"
    assert raw["request_attempts"] == 1 and raw["latency_seconds"] >= 0
    assert raw["system_fingerprint"] == "fp"
    assert raw["usage"] == {"prompt_tokens": 12, "completion_tokens": 7,
                             "total_tokens": None if provider == "anthropic" else 19}
    if provider == "anthropic":
        assert raw["finish_reason"] == "max_tokens"
        assert raw["provider_usage"]["cache_read_input_tokens"] == 3
        assert calls[0]["system"] == "system one\n\nsystem two"
        assert calls[0]["messages"] == [{"role": "user", "content": "question"}]
    else:
        assert raw["finish_reason"] == "length"
        assert raw["provider_usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
    json.dumps(raw)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_old_minimal_response_fixtures_report_unknown_metadata(monkeypatch, provider):
    __import__(f"src.teachers.{provider}_client")
    fake_sdk(monkeypatch, provider, [response(provider, metadata=False)])
    result = build_teacher_client(provider, "requested", api_key="offline-test-placeholder", max_retries=0).chat([])
    assert result.text == "first second"
    assert result.raw["model"] is None and result.raw["finish_reason"] is None
    assert result.raw["provider_usage"] is None and result.raw["id"] is None
    assert result.raw["usage"] == {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_wrapper_retry_count_is_explicit_and_sdk_retries_are_disabled(monkeypatch, provider):
    __import__(f"src.teachers.{provider}_client")
    calls, constructors, sleeps = fake_sdk(monkeypatch, provider, [RuntimeError("transient"), response(provider)])
    result = build_teacher_client(provider, "requested", api_key="offline-test-placeholder", max_retries=1).chat([])
    assert len(calls) == 2 and sleeps == [1] and constructors[0]["max_retries"] == 0
    assert result.raw["request_attempts"] == 2


@pytest.mark.parametrize("provider", PROVIDERS)
def test_final_error_does_not_echo_provider_secrets_or_prompts(monkeypatch, provider):
    __import__(f"src.teachers.{provider}_client")
    error = RuntimeError("Authorization: Bearer DO_NOT_LOG_ME prompt=private question")
    error.status_code = 429
    calls, _, sleeps = fake_sdk(monkeypatch, provider, [error])
    client = build_teacher_client(provider, "requested", api_key="offline-test-placeholder", max_retries=0)
    with pytest.raises(RuntimeError) as caught:
        client.chat([])
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert "DO_NOT_LOG_ME" not in rendered and "private question" not in rendered
    assert "RuntimeError, HTTP 429" in str(caught.value)
    assert len(calls) == 1 and sleeps == []


def test_openai_token_parameter_fallback_is_counted_and_remembered(monkeypatch):
    import src.teachers.openai_client  # noqa: F401
    calls, _, sleeps = fake_sdk(monkeypatch, "openai", [
        RuntimeError("Use max_completion_tokens instead of max_tokens"), response("openai"), response("openai")])
    client = build_teacher_client("openai", "requested", api_key="offline-test-placeholder", max_retries=0)
    first = client.chat([], max_tokens=17)
    second = client.chat([], max_tokens=29)
    assert first.raw["request_attempts"] == 2 and second.raw["request_attempts"] == 1
    assert "max_tokens" in calls[0] and calls[1]["max_completion_tokens"] == 17
    assert "max_tokens" not in calls[1] and calls[2]["max_completion_tokens"] == 29
    assert sleeps == []


def test_openai_does_not_repeat_token_parameter_fallback_after_switch(monkeypatch):
    import src.teachers.openai_client  # noqa: F401
    complaint = RuntimeError("max_tokens max_completion_tokens rejected")
    calls, _, _ = fake_sdk(monkeypatch, "openai", [complaint, complaint, complaint])
    client = build_teacher_client("openai", "requested", api_key="offline-test-placeholder", max_retries=1)
    with pytest.raises(RuntimeError, match="after 3 request"):
        client.chat([])
    assert len(calls) == 3
