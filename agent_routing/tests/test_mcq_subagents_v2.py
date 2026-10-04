"""scripts/train_subagents_v2.py (held-out filtering, grouped validation split, best-epoch selection) and the
pure helpers of scripts/mcq_advisor_ab.py (answer parsing, advisor format checks, sign test)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2 = _load("train_subagents_v2")
AB = _load("mcq_advisor_ab")


def _rows(n):
    return [{"example_id": i, "question_hash": f"q{i // 2}", "prompt": [{"role": "user", "content": str(i)}],
             "response": "{}"} for i in range(n)]


def test_split_drops_held_out_by_id_or_hash_and_groups_questions():
    rows = _rows(40)
    train, val, info = V2.split_rows(rows, {"dev": {("hash", "q0")}, "test": {("id", 2), ("hash", "q2")}}, 0.15, 42)
    assert info["dropped_held_out"] == {"dev": 2, "test": 3} and info["rows_kept"] == 35
    assert not ({r["question_hash"] for r in train} & {r["question_hash"] for r in val})
    assert {r["split"] for r in train} == {"train"} and {r["split"] for r in val} == {"dev"}
    assert V2.split_rows(rows, {}, 0.15, 42)[1] == V2.split_rows(rows, {}, 0.15, 42)[1]
    assert V2.split_rows(rows, {}, 0.15, 7)[1] != V2.split_rows(rows, {}, 0.15, 42)[1]
    from src.subagents.train import validate_sft_splits
    validate_sft_splits(train, val)  # the trainer's own leakage check accepts it


def test_runaway_teacher_responses_are_dropped():
    rows = _rows(10)
    rows[3]["response"] = '{"key_evidence": [], "extracted_facts": ["x = 110.5 / (227.5 + 18n) => 110.5 ='  # cut off
    rows[4]["response"] = "```json\n{\"a\": 1}\n```"
    train, val, info = V2.split_rows(rows, {}, 0.2, 42)
    assert info["dropped_not_json"] == 1 and info["rows_kept"] == 9
    assert 3 not in {r["example_id"] for r in train + val} and 4 in {r["example_id"] for r in train + val}
    assert V2.split_rows(rows, {}, 0.2, 42, require_json=False)[2]["rows_kept"] == 10


def test_best_epoch_is_the_lowest_validation_loss(tmp_path):
    for step in (64, 128, 192):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "adapter_model.safetensors").write_text("w")
    log = [{"epoch": 0.5, "loss": 0.4, "step": 32}, {"epoch": 1.0, "eval_loss": 0.30, "step": 64},
           {"epoch": 2.0, "eval_loss": 0.27, "step": 128}, {"epoch": 3.0, "eval_loss": 0.29, "step": 192}]
    (tmp_path / "checkpoint-192" / "trainer_state.json").write_text(json.dumps({"log_history": log}))
    (tmp_path / "checkpoint-64" / "trainer_state.json").write_text(json.dumps({"log_history": log[:2]}))
    sel = V2.select_best(tmp_path)
    assert sel["best"] == {"epoch": 2.0, "step": 128, "eval_loss": 0.27}
    assert sel["best_checkpoint"].endswith("checkpoint-128") and len(sel["eval"]) == 3
    (tmp_path / "checkpoint-192" / "trainer_state.json").write_text(json.dumps({"log_history": log[:1]}))
    with pytest.raises(RuntimeError, match="no validation losses"):
        V2.select_best(tmp_path)


def test_lora_targets_match_the_paper_era_advisors():
    assert sorted(V2.TARGETS) == sorted(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def test_ab_answer_parsing_format_checks_and_sign_test():
    keys = {"A": 1, "B": 1, "C": 1, "D": 1}
    assert AB.letter("reasoning ANSWER: B\nlater ANSWER: (C)", keys) == "C"
    assert AB.letter("ANSWER: E", keys) is None and AB.letter("", keys) is None
    reasoner = {"case_facts": [], "task_type": "x", "decision_factors": [], "knowledge_slots": [],
                "candidate_considerations": [{"choice_key": k} for k in "ABCD"], "missing_information": [],
                "format_confidence": 0.9}
    assert AB.format_checks("reasoner", "```json\n" + json.dumps(reasoner) + "\n```", keys) == \
        {"json": True, "schema": True, "all_choices": True}
    reasoner["candidate_considerations"] = reasoner["candidate_considerations"][:3]
    assert AB.format_checks("reasoner", json.dumps(reasoner), keys)["all_choices"] is False
    assert AB.format_checks("verifier", '{"checks": [] ', keys) == {"json": False, "schema": False}
    assert AB.sign_test(0, 0) == 1.0 and abs(AB.sign_test(10, 2) - 0.0386) < 1e-3
    assert AB.sign_test(5, 5) == 1.0
