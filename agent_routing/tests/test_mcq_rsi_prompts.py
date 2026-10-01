import hashlib
import json
from pathlib import Path

import pytest

from src.manager.marginal_value import ADVISOR_KINDS
from src.manager.mcq_rsi import prompts
from src.manager.mcq_rsi.benchmarks import BENCHMARKS
from src.subagents.prompts import runtime_prompts
from src.subagents.prompts.runtime_prompts import build_runtime_messages

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mcq_rsi"


def test_prompt_files_match_registry_bytes():
    prompts.validate()
    for bench in BENCHMARKS:
        for kind in ADVISOR_KINDS:
            data = prompts.prompt_path(bench, kind).read_bytes()
            assert hashlib.sha256(data).hexdigest() == prompts.prompt_sha256(bench, kind)
            assert data.endswith(b"\n") and b"\r" not in data
            entry = prompts.PROMPTS[bench][kind]
            assert entry["status"] == "sft_data" and entry["runtime_confirmed"] is False
            assert entry["source"].endswith(".jsonl")


def test_prompt_families():
    text = {(b, k): prompts.system_prompt(b, k) for b in BENCHMARKS for k in ADVISOR_KINDS}
    for kind in ("extractor", "verifier"):
        assert len({text[(b, kind)] for b in BENCHMARKS}) == 1
    assert text[("mmlu_pro", "reasoner")] == text[("gpqa", "reasoner")]
    assert text[("medqa", "reasoner")].startswith("You are the Reasoner sub-agent for medical multiple-choice questions.")
    assert text[("gpqa", "reasoner")].startswith("You are the Reasoner sub-agent for academic multiple-choice questions")
    assert text[("aqua", "reasoner")].startswith("You are the Reasoner sub-agent for five-choice algebraic word problems.")
    # 9.30 kept the Verifier prompt but changed the Extractor and Reasoner ones after the paper.
    assert text[("medqa", "verifier")] == runtime_prompts.VERIFIER_RUNTIME_SYSTEM
    assert text[("medqa", "extractor")] != runtime_prompts.EXTRACTOR_RUNTIME_SYSTEM
    assert all(text[(b, "reasoner")] != runtime_prompts.REASONER_RUNTIME_SYSTEM for b in BENCHMARKS)
    assert [b for b in BENCHMARKS if prompts.PROMPTS[b]["reasoner"]["repo_commit"] is None] == ["mmlu_pro", "gpqa", "aqua"]


def test_tampered_prompt_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "medqa").mkdir()
    (tmp_path / "medqa" / "reasoner.txt").write_text("You are the Reasoner sub-agent.\n")
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    with pytest.raises(ValueError):
        prompts.system_prompt("medqa", "reasoner")
    with pytest.raises(ValueError):
        prompts.prompt_path("medqa", "planner")


def test_build_advisor_messages_replaces_only_the_system_turn():
    choices = {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}
    for kind in ADVISOR_KINDS:
        msgs = prompts.build_advisor_messages("aqua", kind, "What is 1+1?", "", choices, candidate_answer="B")
        base = build_runtime_messages(kind, "What is 1+1?", "", choices, candidate_answer="B")
        assert msgs[0] == {"role": "system", "content": prompts.system_prompt("aqua", kind)}
        assert msgs[1:] == base[1:]
    verifier = prompts.build_advisor_messages("aqua", "verifier", "q", "", choices, candidate_answer="B")
    assert "CANDIDATE ANSWER TO AUDIT: B" in verifier[1]["content"]


def _rows(rel):
    path = ROOT / rel
    if not path.exists():
        return None
    return {r["example_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


@pytest.mark.parametrize("bench", list(BENCHMARKS))
def test_user_messages_match_advisor_sft_data(bench):
    rows = _rows(BENCHMARKS[bench].cache)
    if rows is None:
        pytest.skip(f"{BENCHMARKS[bench].cache} not built")
    fixture = json.loads((FIXTURES / "advisor_user_messages.json").read_text())[bench]
    for kind in ADVISOR_KINDS:
        for item in fixture[kind]:
            row = rows[item["example_id"]]
            msgs = prompts.build_advisor_messages(bench, kind, row["question"], row["context"], row["choices"],
                                                  candidate_answer=item["candidate"])
            assert hashlib.sha256(msgs[1]["content"].encode()).hexdigest() == item["user_sha256"], (kind, item)
        assert any(item["candidate"] for item in fixture["verifier"])
