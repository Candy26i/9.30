import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.verifiable.backend import HTTPAdvisors
from src.verifiable.sampling import normalize_generation, generation_kwargs
from src.verifiable.serve import generate_advisor_request

PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
          "presence_penalty": 1.5, "repetition_penalty": 1.0, "seed": 42}


def test_client_server_roundtrip_settings_cache_and_unmodified_output():
    seen = []
    class Backend:
        def generate(self, messages, max_tokens, generation_options):
            seen.append((messages, max_tokens, generation_options))
            return {"text": "Verdict: incorrect\nLater: actually correct",
                    "prompt_tokens": 20, "completion_tokens": 10, "truncated": False}
    def post(url, json, timeout):
        result, settings = generate_advisor_request(Backend(), json)
        data = {"choices": [{"message": {"content": result["text"]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
                "margent_generation": settings}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    advisor = HTTPAdvisors("http://fake", max_tokens=4096, generation_options=PRESET)
    row = SimpleNamespace(question="Saved question", context="", ground_truth="SECRET")
    with patch("requests.post", side_effect=post):
        first = advisor.call("verifier", row, "saved draft")
        second = advisor.call("verifier", row, "saved draft")
        assert not first["cache_hit"] and second["cache_hit"]
        assert len(seen) == 1
        assert seen[0][1:] == (4096, PRESET)
        assert "SECRET" not in str(seen[0][0])
        assert first["text"] == "Verdict: incorrect\nLater: actually correct"
        advisor.generation_options = {**PRESET, "seed": 43}
        advisor.call("verifier", row, "saved draft")
        assert len(seen) == 2
        assert seen[1][2]["seed"] == 43


def test_sampled_client_rejects_old_server_ignoring_settings():
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {})
    with patch("requests.post", return_value=response):
        with pytest.raises(RuntimeError, match="confirm requested generation"):
            HTTPAdvisors("http://fake", generation_options=PRESET).call(
                "reasoner", SimpleNamespace(question="q", context=""))


def test_sampled_truncation_is_recorded_without_aborting_the_benchmark():
    data = {"choices": [{"message": {"content": "unfinished"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4096}, "margent_generation": PRESET}
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    with patch("requests.post", return_value=response):
        result = HTTPAdvisors("http://fake", max_tokens=4096, generation_options=PRESET).call(
            "reasoner", SimpleNamespace(question="q", context=""))
    assert result["truncated"] and not result["valid_output"]
    assert result["error"] == "advisor_output_truncated"
    assert result["text"] == "unfinished"


@pytest.mark.parametrize("settings", [{"seed": -1}, {"seed": 1.5}, {"temperature": float("nan")},
    {"temperature": -0.1}, {"top_p": 0}, {"top_p": 1.1}, {"top_k": True},
    {"presence_penalty": 3}, {"repetition_penalty": 0}, {"unknown": 1}, "wrong-type"])
def test_invalid_settings_fail_before_inference(settings):
    with pytest.raises(ValueError):
        normalize_generation(settings)


def test_greedy_default_unchanged_and_sampled_options_match_replay(monkeypatch):
    assert normalize_generation() == {"temperature": 0.0, "seed": 42}
    assert generation_kwargs({}, 10) == {"do_sample": False}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(LogitsProcessorList=list))
    opts = generation_kwargs(PRESET, 10)
    assert opts["do_sample"]
    assert opts["temperature"] == 0.7 and opts["top_p"] == 0.8 and opts["top_k"] == 20
    assert opts["min_p"] == 0.0 and opts["repetition_penalty"] == 1.0
    assert len(opts["logits_processor"]) == 1
    assert "seed" not in opts
