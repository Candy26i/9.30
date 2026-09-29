"""Audit regressions: budget-limited model output versus infrastructure failure."""
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.verifiable.answers import correct, extract_final
from src.verifiable.backend import ContextBudgetExceeded, HTTPAdvisors
from src.verifiable.experiment import policy_rollout, collect_one
from src.verifiable.runner import run_data, load_config, validate_resume_records
from src.verifiable.serve import generate_advisor_request
from src.utils.io import write_jsonl
from test_verifiable import Advisors, Backend, CFG, row


def test_legacy_backend_new_constructor_keeps_manager_usage_default(tmp_path):
    from test_rsi import tiny_checkpoint
    from src.verifiable.backend import HFBackend, load_model
    base, checkpoint = tiny_checkpoint(tmp_path)
    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer, backend.model = load_model(str(base), str(checkpoint))
    backend.max_context, backend.decision_constraint = 4096, "none"
    assert "usage_actor" not in backend.__dict__
    with patch("src.verifiable.backend.usage") as log:
        backend.generate([{"role": "user", "content": "one"}], max_tokens=1)
    assert log.call_args.args[0] == "manager"


@pytest.mark.parametrize("value", [
    r"FINAL_ANSWER: $\boxed{42}$.",
    "Reasoning.\nFINAL_ANSWER:\n\\[\\boxed{42}\\]",
    r"**FINAL_ANSWER:** \boxed{42}",
    r"**FINAL_ANSWER: \boxed{42}**",
    r"FINAL_ANSWER: **\boxed{42}**.",
    r"FINAL_ANSWER: \boxed {42}。",
    r"FINAL_ANSWER: \(\boxed{\frac{84}{2}}\)",
])
def test_final_declaration_accepts_presentation_without_changing_math(value):
    assert extract_final(value) is not None
    assert correct(value, "42")
    assert not correct(value, "41")


@pytest.mark.parametrize("value", [
    r"The working contains \boxed{42}.",
    r"FINAL_ANSWER: 42",
    r"FINAL_ANSWER: $\boxed{42}$ or 41",
    "FINAL_ANSWER: \\boxed{42}\nActually 41",
    "FINAL_ANSWER: \\boxed{41}\n**FINAL_ANSWER:** \\boxed{42}",
    r"FINAL_ANSWER: \boxed{42",
    r"FINAL_ANSWER: \boxed{42}\boxed{41}",
])
def test_extraction_never_guesses_a_missing_or_ambiguous_answer(value):
    assert extract_final(value) is None
    assert not correct(value, "42")


def advisor_response(text="partial advice", finish="length", tokens=8):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": tokens}}


def response(data):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)


def test_truncated_advice_stays_raw_cached_and_auditable():
    advisor = HTTPAdvisors("http://fake", max_tokens=8)
    with patch("requests.post", return_value=response(advisor_response())) as post, \
            patch("src.verifiable.backend.generation") as log:
        first = advisor.call("reasoner", row())
        second = advisor.call("reasoner", row())
    assert first["text"] == second["text"] == "partial advice"
    assert first["truncated"] and first["error"] == "advisor_output_truncated"
    assert first["finish_reason"] == "length" and not first["valid_output"]
    assert second["actual_completion_tokens"] == 0 and second["cache_hit"]
    assert post.call_count == 1
    assert [call.kwargs["error"] for call in log.call_args_list] == ["advisor_output_truncated"] * 2


def test_empty_model_advice_is_not_a_transport_failure():
    advisor = HTTPAdvisors("http://fake", max_tokens=8)
    with patch("requests.post", return_value=response(advisor_response("", "stop", 1))):
        result = advisor.call("reasoner", row())
    assert result["text"] == "" and not result["valid_output"]
    assert result["error"] == "advisor_empty_output" and not result["truncated"]


@pytest.mark.parametrize("change", [
    {"usage": {"prompt_tokens": 10, "completion_tokens": -1}},
    {"usage": {"prompt_tokens": 10, "completion_tokens": 9}},
    {"usage": {"prompt_tokens": 10, "completion_tokens": 1.5}},
    {"choices": []},
    {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
    {"choices": [{"message": {"content": "filtered"}, "finish_reason": "content_filter"}]},
])
def test_malformed_advisor_response_does_not_become_a_math_failure(change):
    advisor = HTTPAdvisors("http://fake", max_tokens=8)
    with patch("requests.post", return_value=response({**advisor_response(), **change})):
        with pytest.raises(RuntimeError, match="Malformed advisor response"):
            advisor.call("reasoner", row())


def test_advisor_identity_failure_is_not_relaxed():
    advisor = HTTPAdvisors("http://fake", max_tokens=8)
    advisor.identity = {"model": "frozen"}
    with patch("requests.post", return_value=response({**advisor_response(), "margent_advisor": {"model": "other"}})):
        with pytest.raises(RuntimeError, match="identity changed"):
            advisor.call("reasoner", row())


def test_full_evaluation_continues_after_truncated_advisor(tmp_path, monkeypatch):
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    data, out = tmp_path / "test.jsonl", tmp_path / "eval"
    write_jsonl(str(data), [row(i, split="test").to_dict() for i in range(2)])
    advisors = HTTPAdvisors("http://fake", max_tokens=8)
    with patch("requests.post", return_value=response(advisor_response())):
        result = run_data(CFG, str(data), "fake", out, "evaluate", backend=Backend(), advisors=advisors)
    saved = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]
    assert result["n"] == len(saved) == 2
    assert all(rec["policy"]["correct"] for rec in saved)
    assert all(any(c.get("error") == "advisor_output_truncated" for c in rec["costs"]) for rec in saved)


def test_truncated_manager_answer_is_not_relabelled_as_valid():
    class Truncated(Backend):
        def generate(self, history, tools=None, **kwargs):
            value = super().generate(history, tools=tools, **kwargs)
            if tools:
                value["text"] = "COMMIT"
            else:
                value.update(text=r"FINAL_ANSWER: \boxed{42}", truncated=True)
            return value
    result = policy_rollout(row(), Truncated(), None, CFG, 42)
    assert not result["valid"] and not result["correct"]


def test_context_budget_is_a_sample_failure_and_keeps_zero_actual_cost():
    class TooLong:
        def generate(self, *args, **kwargs):
            raise ContextBudgetExceeded(99, 8, 100)
    result = policy_rollout(row(), TooLong(), None, CFG, 42)
    assert not result["valid"] and not result["correct"]
    assert result["error"] == "context_budget_exceeded"
    generated, settings = generate_advisor_request(TooLong(), {"messages": [], "max_tokens": 8})
    assert generated["completion_tokens"] == generated["actual_prompt_tokens"] == 0
    data = {"choices": [{"message": {"content": generated["text"]}, "finish_reason": "stop"}],
            "usage": {k: generated[k] for k in ("prompt_tokens", "completion_tokens")},
            "margent_generation_error": generated["error"], "margent_max_context": generated["max_context"]}
    with patch("requests.post", return_value=response(data)):
        advice = HTTPAdvisors("http://fake", max_tokens=8).call("reasoner", row())
    assert advice["error"] == "advisor_context_budget_exceeded"
    assert advice["actual_prompt_tokens"] == advice["completion_tokens"] == 0
    assert advice["prompt_tokens"] == 99


@pytest.mark.parametrize("change", [
    {"direct_correct": "false"}, {"direct_valid": None}, {"direct_text": None},
    {"ground_truth": "tampered"}, {"split": "test"}, {"example_id": 900},
    {"costs": []}, {"branches": []}, {"branches": [{"correct": True}]},
])
def test_resume_never_counts_incomplete_or_foreign_shards(change):
    record = collect_one(row(), Backend(), Advisors(), CFG, 42)
    with pytest.raises(ValueError, match="[Rr]esume"):
        validate_resume_records([{**record, **change}], [row()], "collect")


def test_resume_rejects_invalid_shard_before_overwriting_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    data, out = tmp_path / "test.jsonl", tmp_path / "eval"
    write_jsonl(str(data), [row(split="test").to_dict()])
    run_data(CFG, str(data), "fake", out, "evaluate", backend=Backend(), advisors=Advisors())
    ledger = (out / "records.jsonl").read_bytes()
    shard = next((out / "questions").glob("*.json"))
    saved = json.loads(shard.read_text())
    del saved["policy"]
    shard.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="resume outcome"):
        run_data(CFG, str(data), "fake", out, "evaluate", resume=True, backend=Backend(), advisors=Advisors())
    assert (out / "records.jsonl").read_bytes() == ledger


@pytest.mark.parametrize("change", [{"max_depth": True}, {"max_new_tokens": 1.5},
    {"decision_max_tokens": True}, {"advisor_max_tokens": "2048"}, {"generation_seed": -1}])
def test_invalid_generation_configuration_fails_before_model_load(tmp_path, change):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**CFG, **change}))
    with pytest.raises(ValueError):
        load_config(path)
