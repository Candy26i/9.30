import json
import weakref
from pathlib import Path
from unittest.mock import patch

import pytest

from src.verifiable.expert_eval import (BudgetExpired, build_report, evaluate, parse_verdict,
                                        persistent_budget)
from src.verifiable.protocol import KINDS
from src.verifiable.serve import EXPERT_ADAPTERS
from test_expert_serving import write_bundle


@pytest.mark.parametrize("text,expected", [
    ("Verdict: correct\nEvidence: okay", "correct"),
    ("**Verdict**: **INCORRECT**.\nCorrection: 2", "incorrect"),
    ("Verdict: uncertain", "uncertain"),
    ("Verdict: correct\nVerdict: incorrect", None),
    ("This may be incorrect", None), ("Verdict: correct or incorrect", None)])
def test_explicit_verdict_parser(text, expected):
    assert parse_verdict(text) == expected


def rows_for(role):
    return [{"role": role, "split": "dev", "question": f"q{index}", "question_hash": f"hash{index}",
             "context": "", "candidate": "1+1=2" if role == "verifier" else "",
             "prompt": [{"role": "user", "content": f"{role} q{index}"}],
             "response": "SECRET_REFERENCE_ONLY_FOR_HUMANS", "verdict": label,
             "quality_checks": {"scope": "local_arithmetic_step"}, "source_id": f"id{index}"}
            for index, label in enumerate(("correct", "incorrect"))]


def result(text="hint", truncated=False, role="verifier"):
    return {"text": text, "truncated": truncated, "prompt_tokens": 4,
            "completion_tokens": 2, "seconds": 0.01, "actual_role": role}


def test_report_keeps_unparsed_in_denominator_and_scores_only_observed_truth_classes():
    selected = {role: rows_for(role) for role in KINDS}
    values = {("prompt_only", "verifier", 0): result("Verdict: correct", truncated=True),
              ("prompt_only", "verifier", 1): result("I do not know"),
              ("trained", "verifier", 0): result("Verdict: uncertain"),
              ("trained", "verifier", 1): result("Verdict: incorrect")}
    report = build_report(selected, values)
    stats = report["arms"]["prompt_only"]["verifier"]
    assert stats["n_completed"] == 2 and stats["unparsed_count"] == 1
    assert stats["truncated_count"] == 1 and stats["verdict_accuracy"] == .5
    assert stats["labeled_classes"] == ["correct", "incorrect"]
    assert stats["macro_f1"] == .5
    assert report["arms"]["trained"]["verifier"]["confusion"]["correct"]["uncertain"] == 1
    assert not report["arms"]["trained"]["extractor"]["quality_assessed"]
    assert report["completed_generations"] == 4 and report["expected_generations"] == 12
    assert not report["evaluation_complete"] and not report["independent_benchmark"]


def setup_eval(tmp_path, monkeypatch, fail_at=None):
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    bundle, _ = write_bundle(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text("{}")
    monkeypatch.setattr("src.verifiable.expert_eval.read_dataset", lambda root, role: ({}, [], rows_for(role)))
    live = weakref.WeakSet()
    calls, loaded = [], []
    class Backend:
        def __init__(self, *args, **kwargs):
            assert len(live) == 0, "Two complete models loaded at once"
            live.add(self)
            self.arm = "trained" if isinstance(args[0], dict) else "prompt_only"
            self.bundle = args[0] if self.arm == "trained" else None
            loaded.append(self.arm)
    def generate(backend, request):
        assert "SECRET_REFERENCE" not in str(request)
        if fail_at is not None and len(calls) == fail_at:
            raise RuntimeError("simulated interrupted generation")
        calls.append((backend.arm, request["model"], request["messages"]))
        role = request["model"]
        value = result("Verdict: correct" if role == "verifier" else "grounded hint", role=role)
        if backend.bundle:
            value["advisor_role"] = {"role": role, "adapter_name": EXPERT_ADAPTERS[role],
                                     "identity": backend.bundle["roles"][role]["identity"]}
        return value, {"temperature": 0., "seed": 42}
    monkeypatch.setattr("src.verifiable.expert_eval.HFBackend", Backend)
    monkeypatch.setattr("src.verifiable.expert_eval.FrozenExpertBackend", Backend)
    monkeypatch.setattr("src.verifiable.expert_eval.generate_advisor_request", generate)
    return bundle, data, calls, loaded


def test_eval_pairs_outputs_releases_base_before_experts_and_resumes_without_inference(tmp_path, monkeypatch):
    bundle, data, calls, loaded = setup_eval(tmp_path, monkeypatch)
    out = tmp_path / "evaluation"
    report = evaluate(bundle, data, out, limit=2, minutes=1)
    assert report["evaluation_complete"] and len(calls) == 12
    assert loaded == ["prompt_only", "trained"]
    review = [json.loads(line) for line in (out / "review.jsonl").read_text().splitlines()]
    assert len(review) == 6 and all(set(row["outputs"]) == {"prompt_only", "trained"} for row in review)
    assert all(row["human_review"]["status"] == "pending" for row in review)
    assert all(row["reference_is_model_input"] is False for row in review)
    journal = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines()]
    assert len(journal) == 12 and all(r["actual_role"] in KINDS and "advisor_role" in r for r in journal)
    again = evaluate(bundle, data, out, limit=2, minutes=1)
    assert again["evaluation_complete"] and len(calls) == 12 and len(loaded) == 2
    with pytest.raises(ValueError, match="configuration/data/bundle changed"):
        evaluate(bundle, data, out, limit=1, minutes=1)


def test_failed_attempt_retains_completed_shards_and_partial_review(tmp_path, monkeypatch):
    bundle, data, calls, _ = setup_eval(tmp_path, monkeypatch, fail_at=3)
    out = tmp_path / "evaluation"
    with pytest.raises(RuntimeError, match="simulated"):
        evaluate(bundle, data, out, limit=2, minutes=1)
    report = json.loads((out / "report.json").read_text())
    assert report["controller_status"] == "failed" and report["completed_generations"] == 3
    assert len(list((out / "shards").rglob("*.json"))) == 3
    assert (out / "review.jsonl").exists()
    original = __import__("src.verifiable.expert_eval", fromlist=["generate_advisor_request"]).generate_advisor_request
    def working(backend, request):
        calls.append((backend.arm, request["model"], request["messages"]))
        role = request["model"]
        value = result("hint", role=role)
        if backend.bundle:
            value["advisor_role"] = {"role": role, "adapter_name": EXPERT_ADAPTERS[role],
                                     "identity": backend.bundle["roles"][role]["identity"]}
        return value, {}
    monkeypatch.setattr("src.verifiable.expert_eval.generate_advisor_request", working)
    # The failed backend was disposed before the next attempt (including its traceback).
    report = evaluate(bundle, data, out, limit=2, minutes=1)
    assert report["evaluation_complete"] and len(calls) == 12
    assert original is not working


def test_malformed_completed_shard_cannot_skip_generation(tmp_path, monkeypatch):
    bundle, data, _, _ = setup_eval(tmp_path, monkeypatch)
    out = tmp_path / "evaluation"
    evaluate(bundle, data, out, limit=1, minutes=1)
    path = next((out / "shards").rglob("*.json"))
    saved = json.loads(path.read_text())
    saved["result"].pop("text")
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate(bundle, data, out, limit=1, minutes=1)


def test_persistent_budget_cannot_reset_after_crash_or_change_limit(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text(json.dumps({"minutes": 1, "spent_seconds": 40, "active_started_unix": 1000}))
    with patch("src.verifiable.expert_eval.time.time", return_value=1030):
        with pytest.raises(BudgetExpired), persistent_budget(tmp_path, 1):
            pass
    assert json.loads(path.read_text())["spent_seconds"] == 70
    with pytest.raises(ValueError, match="budget changed"), persistent_budget(tmp_path, 2):
        pass


def test_exhausted_eval_returns_incomplete_report_without_loading_model(tmp_path, monkeypatch):
    bundle, data, calls, loaded = setup_eval(tmp_path, monkeypatch)
    out = tmp_path / "evaluation"
    # Finish a clean manifest then remove one shard to represent missing work.
    evaluate(bundle, data, out, limit=1, minutes=1)
    next((out / "shards").rglob("*.json")).unlink()
    (out / "budget.json").write_text(json.dumps({"minutes": 1, "spent_seconds": 60, "active_started_unix": None}))
    n, models = len(calls), len(loaded)
    report = evaluate(bundle, data, out, limit=1, minutes=1)
    assert not report["evaluation_complete"] and report["controller_status"] == "budget_exhausted"
    assert len(calls) == n and len(loaded) == models
