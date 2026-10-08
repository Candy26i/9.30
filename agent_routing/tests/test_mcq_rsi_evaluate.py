"""MCQ RSI evaluation (design §3.6, §5): stages pool/out_dir injection, fail-stop, metrics, gates."""
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from mcq_rsi_helpers import advisor_server, make_rows, tiny_tokenizer
from src.manager.mcq_rsi import evaluate as E
from src.manager.mcq_rsi.advisors import AdvisorFailStop, CachedAdvisorPool
from src.pipeline import stages

ROOT = Path(__file__).resolve().parents[1]
STAGES_BEFORE_PR3 = "c09f429baf4bbabb36f9805b71e933a545b6aea9"  # git blob of src/pipeline/stages.py before this PR
OFFSET = 10_000_000
REAL_RESOLVE_BASE = E.resolve_base  # the autouse ``pinned_base`` fixture replaces it (no downloads)

CALL = "DRAFT_ANSWER_{d}\n\n<tool_call>\n<function={tool}>\n{params}</function>\n</tool_call>"


def call(draft, tool, current=None):
    params = f"<parameter=current_draft>\n{current}\n</parameter>\n" if current else ""
    return CALL.format(d=draft, tool=tool, params=params)


COMMIT = "DRAFT_ANSWER_{0}\nANSWER_{0}".format
# make_rows(4): ids 100-103, gold A, B, C, D.
SCRIPT = {
    100: [COMMIT("A")],  # correct draft, commits
    101: [call("A", "verifier_tool", "A"), COMMIT("B")],  # wrong draft, corrected by the Verifier
    102: [call("C", "extractor_tool"), COMMIT("D")],  # correct draft, corrupted
    103: [call("B", "reasoner_tool"), call("B", "verifier_tool", "B"), COMMIT("B")],  # wrong, stays wrong
}


class ScriptTok:
    """The tiny tokenizer (real chat template) whose ``decode`` returns scripted turns (``<tool_call>`` kept, as Qwen)."""

    def __init__(self):
        self.tok, self.texts = tiny_tokenizer(), []
        self.pad_token_id, self.eos_token_id = self.tok.pad_token_id, self.tok.eos_token_id

    def apply_chat_template(self, *a, **k):
        return self.tok.apply_chat_template(*a, **k)

    def __call__(self, *a, **k):
        return self.tok(*a, **k)

    def decode(self, ids, skip_special_tokens=False):
        ids = [int(i) for i in ids]
        if len(ids) == 1 and ids[0] >= OFFSET:
            return self.texts[ids[0] - OFFSET]
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


class ScriptModel:
    def __init__(self, tok, script):
        self.tok, self.script, self.prompts = tok, script, []

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, **kw):
        import torch
        assert kw["do_sample"] is False and kw["max_new_tokens"] in (1024, E.MAX_NEW_TOKENS)
        prompt = self.tok.tok.decode(input_ids[0].tolist())
        self.prompts.append(prompt)
        eid = int(re.search(r"Example ID: (\d+)", prompt).group(1))
        turn = prompt.count("<tool_response>")
        self.tok.texts.append(self.script[eid][min(turn, len(self.script[eid]) - 1)])
        return torch.cat([input_ids, torch.tensor([[OFFSET + len(self.tok.texts) - 1]])], 1)


@pytest.fixture(autouse=True)
def pinned_base(tmp_path, monkeypatch):
    """``resolve_base`` offline: a local HF-cache snapshot of BASE_MODEL@BASE_REVISION (no download)."""
    from src.manager.mcq_rsi import benchmarks as registry
    snapshot = (tmp_path / "hf" / ("models--" + registry.BASE_MODEL.replace("/", "--")) / "snapshots"
                / registry.BASE_REVISION)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    calls = []

    def resolve(model, revision):
        calls.append((model, revision))
        if Path(model).is_dir() or not revision:
            return model
        assert (model, revision) == (registry.BASE_MODEL, registry.BASE_REVISION)
        return str(snapshot)

    monkeypatch.setattr(E, "resolve_base", resolve)
    return types.SimpleNamespace(snapshot=snapshot, calls=calls)


@pytest.fixture
def scripted(monkeypatch):
    tok = ScriptTok()
    model = ScriptModel(tok, SCRIPT)
    loads, bases = [], []

    def load(ctx, manager_dir, device, dtype):
        loads.append(manager_dir)
        bases.append(ctx.base_model)
        return tok, model

    monkeypatch.setattr(stages, "_load_manager_for_eval", load)
    return types.SimpleNamespace(tok=tok, model=model, loads=loads, bases=bases)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "ckpt"
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    return path


def make_pool(tmp_path, **kw):
    http, adapters = advisor_server(tmp_path, **kw)
    return CachedAdvisorPool("medqa", tmp_path / "advisor_cache", "http://x", adapters=adapters, http=http,
                             sleep=lambda s: None), http


# --------------------------------------------------------------------------- evaluate

def test_evaluate_injects_pool_and_computes_metrics(tmp_path, scripted, checkpoint):
    pool, http = make_pool(tmp_path)
    rows = make_rows(4)
    result = E.evaluate(checkpoint, rows, pool, tmp_path / "dev", bench="medqa")
    m = result["metrics"]
    assert result["passed"] and result["gate"] == []
    assert m["n"] == 4 and m["accuracy"] == 0.5 and m["initial_draft_accuracy"] == 0.5 and m["gain_pp"] == 0.0
    assert m["calls_per_example"] == 1.0 and m["call_rate"] == 0.75
    assert m["per_advisor_calls"] == {"extractor_tool": 1, "reasoner_tool": 1, "verifier_tool": 2}
    assert m["correction_rate"] == 0.25 and m["corruption_rate"] == 0.25
    assert m["call_rate_given_draft_wrong"] == 1.0 and m["call_rate_given_draft_correct"] == 0.5 and m["call_gap"] == 0.5
    assert m["malformed_tool_calls"] == 0 and m["error_payload_tool_outputs"] == 0
    # Outputs live in the stage directory; the StageContext's own eval_root stays empty.
    out = tmp_path / "dev"
    report = json.loads((out / "manager_tool_eval_report.json").read_text())
    assert report["binding_mode"] == "environment" and report["n_samples"] == 4
    assert not list((out / ".stage_ctx").rglob("manager_tool_eval*"))
    # Two-pass speculative prefetch: pass 1 fetched each calling root's first call plus V(q, X), so the eval
    # itself (including 103's second-turn Verifier call on the unchanged draft B) never waited on a fetch.
    assert result["prefetch"]["first_turn_calls"] == 3 and result["prefetch"]["fetched"] == 5
    assert len(http.posts) == 5 and result["advisor_stats"]["fetched"] == 5
    fields = [json.loads(p.read_text())["fields"] for p in (tmp_path / "advisor_cache").rglob("*.json")]
    assert sorted(f["candidate"] for f in fields if f["kind"] == "verifier") == ["A", "B", "C"]
    # Deployment protocol: environment schema, example_id injected into the kept call turn, task description.
    assert '"example_id"' not in json.dumps(stages._manager_tool_schemas("environment"))
    assert "You are a manager agent solving a medical multiple-choice question." in scripted.model.prompts[0]
    assert scripted.loads == [str(checkpoint), str(checkpoint)]  # pass 1 and pass 2
    # Re-running the stage returns the stored result; other inputs are refused.
    assert E.evaluate(checkpoint, rows, pool, out, bench="medqa") == result
    with pytest.raises(ValueError, match="different inputs"):
        E.evaluate(checkpoint, rows[:3], pool, out, bench="medqa")


def test_evaluate_without_speculation_matches(tmp_path, scripted, checkpoint):
    pool, http = make_pool(tmp_path)
    a = E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "a", bench="medqa", speculative=False)
    assert a["prefetch"] is None and len(http.posts) == 4  # only the 4 calls the manager makes, fetched in line
    b = E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "b", bench="medqa")
    # The one wrong guess: V(102, C), speculated after 102's Extractor call, which 102 never makes.
    assert a["metrics"] == b["metrics"] and b["prefetch"]["fetched"] == 1 and len(http.posts) == 5
    assert (tmp_path / "a" / "manager_tool_eval.jsonl").read_bytes() == (tmp_path / "b" / "manager_tool_eval.jsonl").read_bytes()


def test_fail_stop_aborts_eval_and_pool_is_reusable(tmp_path, scripted, checkpoint):
    broken = {"on": True}
    pool, http = make_pool(tmp_path, fail_when=lambda body: broken["on"] and body["model"] == "medqa_verifier")
    for speculative in (False, True):
        out = tmp_path / f"dev_{speculative}"
        with pytest.raises(AdvisorFailStop):
            E.evaluate(checkpoint, make_rows(4), pool, out, bench="medqa", speculative=speculative)
        assert not (out / "manager_tool_eval.jsonl").exists() and not (out / "mcq_rsi_eval.json").exists()
        assert pool._abort.is_set()
    broken["on"] = False
    result = E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "dev_ok", bench="medqa")  # clear_abort first
    assert result["passed"] and not any("error" in json.dumps(r) for r in
                                        map(json.loads, open(tmp_path / "dev_ok" / "manager_tool_eval.jsonl")))


def test_stages_swallows_errors_without_fail_stop_and_the_gate_catches_it(tmp_path, scripted, checkpoint):
    class Erroring:
        def call(self, agent_kind, **kw):
            if agent_kind == "verifier":
                raise RuntimeError("server down")
            return "fine"

    ctx = E.stage_context("base", tmp_path / "raw")
    report = stages.run_eval_manager_tools(ctx, E.standard_rows(make_rows(4)), str(checkpoint), n_samples=4,
                                           task_description="t", pool=Erroring(), out_dir=str(tmp_path / "raw"))
    records = [json.loads(x) for x in open(tmp_path / "raw" / "manager_tool_eval.jsonl")]
    metrics = E.eval_metrics(records, report)
    assert report["malformed_tool_calls"] == 2 and metrics["error_payload_tool_outputs"] == 2
    gate = E.eval_gate(report, metrics)
    assert any("malformed_tool_calls=2" in g for g in gate) and any("error payloads" in g for g in gate)


def test_gate_failure_raises_after_writing_evidence(tmp_path, monkeypatch, checkpoint):
    tok = ScriptTok()
    script = {**SCRIPT, 100: ["I am not sure."]}  # unparseable final answer -> valid_answer_rate < 1
    monkeypatch.setattr(stages, "_load_manager_for_eval", lambda *a: (tok, ScriptModel(tok, script)))
    pool, _ = make_pool(tmp_path)
    with pytest.raises(E.EvalGateFailed) as err:
        E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "dev", bench="medqa")
    assert err.value.result["gate"] == ["valid_answer_rate=0.75"]
    assert json.loads((tmp_path / "dev" / "mcq_rsi_eval.json").read_text())["passed"] is False
    with pytest.raises(E.EvalGateFailed):  # a rerun does not launder a failed eval
        E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "dev", bench="medqa")
    assert E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "dev", bench="medqa", require_gate=False)["gate"]


def test_evaluate_forced_verifier_gets_no_candidate(tmp_path, scripted, checkpoint):
    scripted.model.script = {eid: [turns[-1]] for eid, turns in SCRIPT.items()}  # forced: one answer turn
    pool, http = make_pool(tmp_path)
    result = E.evaluate_forced(checkpoint, make_rows(4), pool, tmp_path / "forced", ["verifier"], bench="medqa")
    assert result["passed"] and result["metrics"]["calls_per_example"] == 1.0 and result["metrics"]["n"] == 4
    assert (tmp_path / "forced" / "manager_forced_verifier.jsonl").exists()
    assert len(http.posts) == 4 and all(p["model"] == "medqa_verifier" for p in http.posts)
    keys = [json.loads(p.read_text())["fields"] for p in (tmp_path / "advisor_cache").rglob("*.json")]
    assert {k["candidate"] for k in keys} == {""}
    with pytest.raises(ValueError, match="unknown forced"):
        E.evaluate_forced(checkpoint, make_rows(4), pool, tmp_path / "x", ["oracle"], bench="medqa")


# ------------------------------------------------------------ stages defaults unchanged

def _stages_before_pr3():
    try:
        src = subprocess.run(["git", "cat-file", "-p", STAGES_BEFORE_PR3], cwd=ROOT, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if src.returncode != 0:
        return None
    name = "src.pipeline._stages_before_pr3"
    module = types.ModuleType(name)
    module.__package__, module.__file__ = "src.pipeline", str(ROOT / "src/pipeline/_stages_before_pr3.py")
    sys.modules[name] = module
    try:
        exec(compile(src.stdout, module.__file__, "exec"), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


class FakeRemote:
    """``RemoteSubagentPool`` stand-in that records every call's keyword arguments."""
    instances = []
    calls = []

    def __init__(self, server_url=None, **kw):
        self.server_url = server_url
        FakeRemote.instances.append(self)

    def has(self, kind):
        return True

    def call(self, agent_kind, example_id, question, context, choices, cache_namespace="default", candidate_answer=""):
        FakeRemote.calls.append({"agent_kind": agent_kind, "example_id": example_id, "question": question,
                                 "context": context, "choices": dict(choices), "cache_namespace": cache_namespace,
                                 "candidate_answer": candidate_answer})
        return f"{agent_kind} on {example_id} ({cache_namespace}) candidate={candidate_answer or '-'}"


def _run_both(module, tmp_path, tag, monkeypatch, scripted):
    monkeypatch.setattr(module, "_load_manager_for_eval", lambda ctx, d, device, dtype: (scripted.tok, scripted.model))
    ctx = module.StageContext(base_model="base", teacher_id="mcq_default", output_root=str(tmp_path / tag),
                              binding_mode="environment")
    rows = E.standard_rows(make_rows(4))
    tools = module.run_eval_manager_tools(ctx, rows, str(tmp_path / "ckpt"), n_samples=4, task_description="t",
                                          subagent_server_url="http://fake")
    forced = module.run_eval_manager_forced(ctx, rows, str(tmp_path / "ckpt"), forced_tools=["verifier", "reasoner"],
                                            n_samples=4, task_description="t", subagent_server_url="http://fake")
    files = {p.relative_to(tmp_path / tag): p.read_bytes() for p in sorted((tmp_path / tag).rglob("*")) if p.is_file()}
    return tools, forced, files


def test_stages_defaults_are_unchanged(tmp_path, monkeypatch, scripted, checkpoint):
    import src.subagents.runtime as runtime
    monkeypatch.setattr(runtime, "RemoteSubagentPool", FakeRemote)
    monkeypatch.setattr(FakeRemote, "instances", [])
    monkeypatch.setattr(FakeRemote, "calls", [])
    tools, forced, files = _run_both(stages, tmp_path, "new", monkeypatch, scripted)
    # The advisor calls themselves, independent of the pre-PR3 blob: the historical cache namespaces and arguments.
    rows = {r["example_id"]: r for r in make_rows(4)}
    tool_calls = [c for c in FakeRemote.calls if c["cache_namespace"] != "eval_forced"]
    forced_calls = [c for c in FakeRemote.calls if c["cache_namespace"] == "eval_forced"]
    assert sorted((c["example_id"], c["agent_kind"], c["candidate_answer"]) for c in tool_calls) == [
        (101, "verifier", "A"), (102, "extractor", ""), (103, "reasoner", ""), (103, "verifier", "B")]
    assert {c["cache_namespace"] for c in tool_calls} == {"eval_manager_tools"}
    assert sorted((c["example_id"], c["agent_kind"]) for c in forced_calls) == sorted(
        (e, k) for e in rows for k in ("verifier", "reasoner"))
    assert {c["candidate_answer"] for c in forced_calls} == {""}
    assert len(FakeRemote.calls) == 4 + 8
    for c in FakeRemote.calls:
        r = rows[c["example_id"]]
        assert (c["question"], c["context"], c["choices"]) == (r["question"], r["context"], r["choices"])
    # Default parameters: outputs at ctx.eval_root under the historical names, the URL-built pool used.
    assert sorted(map(str, files)) == ["eval/mcq_default/manager_forced_verifier_reasoner.jsonl",
                                       "eval/mcq_default/manager_forced_verifier_reasoner_report.json",
                                       "eval/mcq_default/manager_tool_eval.jsonl",
                                       "eval/mcq_default/manager_tool_eval_report.json"]
    assert FakeRemote.instances and all(r.server_url == "http://fake" for r in FakeRemote.instances)
    assert tools["subagents"] == ["remote"] and forced["k"] == 2
    before = _stages_before_pr3()
    if before is None:
        pytest.skip(f"pre-PR3 stages.py (git blob {STAGES_BEFORE_PR3}) not available")
    old_tools, old_forced, old_files = _run_both(before, tmp_path, "old", monkeypatch, scripted)
    assert (old_tools, old_forced) == (tools, forced)
    assert old_files == files  # byte-identical outputs at identical paths


# -------------------------------------------------------------------------- metrics

def rec(eid, draft, correct, final_correct, calls, names=None, tool_contents=()):
    return {"example_id": eid, "initial_draft": draft, "initial_draft_correct": correct, "correct": final_correct,
            "pred": "A" if final_correct else "B", "valid_answer": True, "tool_calls": calls,
            "tool_names_called": names or [], "corrected_by_tools": bool(calls and not correct and final_correct),
            "corrupted_by_tools": bool(calls and correct and not final_correct),
            "trajectory": [{"role": "tool", "content": c} for c in tool_contents]}


def test_metrics_on_synthetic_records():
    records = [
        rec(1, "A", True, True, 0),
        rec(2, "A", True, False, 1, ["verifier_tool"]),  # corruption
        rec(3, "B", False, True, 2, ["reasoner_tool", "verifier_tool"]),  # correction
        rec(4, "B", False, False, 1, ["extractor_tool"]),
        rec(5, "B", False, False, 0),
        rec(6, None, False, True, 0),  # no draft parsed: outside the conditional metrics
    ]
    m = E.eval_metrics(records)
    assert m["n"] == 6 and m["accuracy"] == pytest.approx(3 / 6)
    assert m["initial_draft_coverage"] == pytest.approx(5 / 6) and m["initial_draft_accuracy"] == pytest.approx(2 / 5)
    assert m["gain_pp"] == pytest.approx(100 * (0.5 - 0.4))
    assert m["calls_per_example"] == pytest.approx(4 / 6) and m["call_rate"] == pytest.approx(3 / 6)
    assert m["per_advisor_calls"] == {"extractor_tool": 1, "reasoner_tool": 1, "verifier_tool": 2}
    assert m["first_call_counts"] == {"extractor_tool": 1, "reasoner_tool": 1, "verifier_tool": 1}
    assert m["correction_rate"] == pytest.approx(1 / 6) and m["corruption_rate"] == pytest.approx(1 / 6)
    assert m["call_rate_given_draft_wrong"] == pytest.approx(2 / 3)
    assert m["call_rate_given_draft_correct"] == pytest.approx(1 / 2)
    assert m["call_gap"] == pytest.approx(2 / 3 - 1 / 2)
    bad = records + [rec(7, "A", True, True, 1, ["verifier_tool"], ['{"error": "timeout"}', "ok"])]
    assert E.eval_metrics(bad)["error_payload_tool_outputs"] == 1
    report = {"accuracy": 0.9, "initial_draft_accuracy": 0.4, "initial_draft_coverage": 5 / 6,
              "avg_tool_calls": 4 / 6, "tool_call_rate": 0.5, "correction_rate": 1 / 6, "corruption_rate": 1 / 6,
              "draft_conditioned_call_gap": 1 / 6, "valid_answer_rate": 1.0, "n_samples": 6, "tool_counts": {}}
    with pytest.raises(ValueError, match="accuracy"):
        E.eval_metrics(records, report)


# ---------------------------------------------------------------------------- gates

def test_eval_gate():
    ok = {"malformed_tool_calls": 0, "valid_answer_rate": 1.0}
    assert E.eval_gate(ok) == []
    assert E.eval_gate({**ok, "malformed_tool_calls": 3}) == ["malformed_tool_calls=3"]
    assert E.eval_gate({**ok, "valid_answer_rate": 0.995}) == ["valid_answer_rate=0.995"]
    assert E.eval_gate({"valid_answer_rate": 1.0}) == ["malformed_tool_calls=None"]  # a report without the count fails
    assert E.eval_gate({"forced_tools": ["verifier"], "valid_answer_rate": 1.0}) == []
    assert E.eval_gate(ok, {"error_payload_tool_outputs": 1}) == ["error payloads in tool outputs=1"]
    assert E.eval_gate(ok, {"error_payload_tool_outputs": 0, "duplicate_tool_calls": 2}) == []


def test_repeated_call_is_reported_not_gated(tmp_path, monkeypatch, checkpoint):
    """stages answers a repeated advisor kind with its own tool_already_called payload: manager behaviour."""
    tok = ScriptTok()
    script = {**SCRIPT, 100: [call("A", "verifier_tool", "A"), call("A", "verifier_tool", "A"), COMMIT("A")]}
    monkeypatch.setattr(stages, "_load_manager_for_eval", lambda *a: (tok, ScriptModel(tok, script)))
    pool, _ = make_pool(tmp_path)
    result = E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "dev", bench="medqa")
    m = result["metrics"]
    assert result["passed"] and result["gate"] == []
    assert m["duplicate_tool_calls"] == 1 and m["error_payload_tool_outputs"] == 0 and m["malformed_tool_calls"] == 0
    records = [json.loads(x) for x in open(tmp_path / "dev" / "manager_tool_eval.jsonl")]
    assert any("tool_already_called" in json.dumps(r) for r in records if r["example_id"] == 100)
    tool = [e for r in records if r["example_id"] == 100 for e in r["trajectory"] if e.get("role") == "tool"]
    assert [e["content"] for e in tool][1] == E.DUPLICATE_PAYLOAD and tool[0]["name"] == tool[1]["name"]


def test_duplicate_payload_vs_genuine_advisor_errors():
    """Only stages' exact payload on a tool that already answered is a duplicate; everything else is gated."""
    def record(*outputs):
        r = rec(1, "A", True, True, len(outputs), [n for n, _ in outputs])
        r["trajectory"] = [{"role": "tool", "name": n, "content": c} for n, c in outputs]
        return r

    ok, dup = "advice", E.DUPLICATE_PAYLOAD
    m = E.eval_metrics([record(("verifier_tool", ok), ("verifier_tool", dup))])
    assert (m["duplicate_tool_calls"], m["error_payload_tool_outputs"]) == (1, 0)
    # Stages dedupes by kind: "verifier" and "verifier_tool" are the same advisor.
    m = E.eval_metrics([record(("verifier", ok), ("verifier_tool", dup))])
    assert (m["duplicate_tool_calls"], m["error_payload_tool_outputs"]) == (1, 0)
    # Genuine advisor errors next to a duplicate still count (and fail the gate).
    m = E.eval_metrics([record(("verifier_tool", ok), ("verifier_tool", dup), ("reasoner_tool", '{"error": "timeout"}'))])
    assert (m["duplicate_tool_calls"], m["error_payload_tool_outputs"]) == (1, 1)
    assert E.eval_gate({"malformed_tool_calls": 0, "valid_answer_rate": 1.0}, m) == ["error payloads in tool outputs=1"]
    # The payload on a tool's first output, or a look-alike error text, is an advisor error, not a duplicate.
    for outputs in ([("verifier_tool", dup)], [("extractor_tool", ok), ("verifier_tool", dup)],
                    [("verifier_tool", ok), ("verifier_tool", '{"error": "tool_already_called"}')]):
        m = E.eval_metrics([record(*outputs)])
        assert (m["duplicate_tool_calls"], m["error_payload_tool_outputs"]) == (0, 1), outputs


def test_base_identity_pins_the_registry_revision(tmp_path, monkeypatch):
    from src.manager.mcq_rsi import benchmarks as registry
    repo = tmp_path / ("models--" + registry.BASE_MODEL.replace("/", "--")) / "snapshots"
    good, bad = repo / registry.BASE_REVISION, repo / ("0" * 40)
    for d in (good, bad):
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
    ident = E.base_identity(str(good))
    assert ident["commit"] == registry.BASE_REVISION and "config.json" in ident["files"]
    with pytest.raises(ValueError, match="pinned revision"):
        E.base_identity(str(bad))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo_id, f: str(bad / f))
    with pytest.raises(ValueError, match="pinned revision"):
        E.base_identity(registry.BASE_MODEL)
    assert E.base_identity("other/model") == {"model_id": "other/model", "commit": "0" * 40}
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo_id, f: str(good / f))
    assert E.base_identity(registry.BASE_MODEL)["commit"] == registry.BASE_REVISION
    assert REAL_RESOLVE_BASE(str(good)) == str(good)  # local directories are used as given
    assert REAL_RESOLVE_BASE("other/model", None) == "other/model"  # unpinned: the hub id as given


def test_grpo_accept_thresholds():
    S = {"accuracy": 0.80, "calls_per_example": 0.40, "call_gap": 0.40}
    edge = {"accuracy": 0.79, "calls_per_example": 0.55, "call_gap": 0.20}  # exactly at every bound
    assert E.grpo_accept(edge, S, True)["accepted"]
    for change, reason in (({"accuracy": 0.7899}, "accuracy"), ({"calls_per_example": 0.5501}, "calls"),
                           ({"call_gap": 0.1999}, "call gap")):
        out = E.grpo_accept({**edge, **change}, S, True)
        assert not out["accepted"] and out["decision"] == "grpo_rejected" and reason in out["reasons"][0]
    assert "uninformative" in E.grpo_accept(edge, S, False)["reasons"][0]
    better = E.grpo_accept({"accuracy": 0.85, "calls_per_example": 0.3, "call_gap": 0.5}, S, True)
    assert better["accepted"] and better["deltas"]["accuracy"] == pytest.approx(0.05)
    # Eval results are accepted too; a failed eval gate rejects.
    assert not E.grpo_accept({"metrics": edge, "gate": ["malformed_tool_calls=1"]}, {"metrics": S, "gate": []}, True)["accepted"]
    assert E.grpo_accept({"metrics": edge, "gate": []}, {"metrics": S, "gate": []}, True)["accepted"]


def test_grpo_accept_negative_sft_gap():
    """With S_k's gap < 0, 0.5 x gap > gap: the bound is gap - 0.5 |gap|, never above S_k's own gap."""
    S = {"accuracy": 0.54, "calls_per_example": 0.5, "call_gap": -0.09}  # the D4 GPQA alternative's dev gap
    for gap in (-0.09, -0.05, 0.1, -0.135):
        assert E.grpo_accept({**S, "call_gap": gap}, S, True)["accepted"], gap
    out = E.grpo_accept({**S, "call_gap": -0.1351}, S, True)
    assert not out["accepted"] and "call gap" in out["reasons"][0]
    # Non-negative S_k gaps keep exactly the 0.5 x S_k bound.
    P = {**S, "call_gap": 0.017}
    assert E.grpo_accept({**P, "call_gap": 0.0085}, P, True)["accepted"]
    assert not E.grpo_accept({**P, "call_gap": 0.0084}, P, True)["accepted"]
    Z = {**S, "call_gap": 0.0}
    assert E.grpo_accept(Z, Z, True)["accepted"] and not E.grpo_accept({**Z, "call_gap": -0.001}, Z, True)["accepted"]
    # G_k = S_k (an informative stage that changed nothing measurable) is never rejected, for any S_k gap sign.
    for gap in (-1.0, -0.5, -0.09, -1e-9, 0.0, 1e-9, 0.017, 0.4, 1.0):
        for acc, calls in ((0.0, 0.0), (0.54, 0.5), (1.0, 3.0)):
            same = {"accuracy": acc, "calls_per_example": calls, "call_gap": gap}
            out = E.grpo_accept(same, dict(same), True)
            assert out["accepted"] and out["decision"] == "grpo_accepted", same
            assert E.grpo_accept({"metrics": same, "gate": []}, {"metrics": same, "gate": []}, True)["accepted"]


def test_sft_flags():
    prev = {"accuracy": 0.80, "calls_per_example": 0.4, "call_gap": 0.4}
    assert E.sft_flags({"accuracy": 0.77, "calls_per_example": 1.5}, prev) == []
    flags = E.sft_flags({"accuracy": 0.7699, "calls_per_example": 1.51}, prev)
    assert len(flags) == 2 and "calls" in flags[0] and "accuracy" in flags[1]


def test_cli_evaluate(tmp_path, scripted, checkpoint, monkeypatch):
    from src.manager.mcq_rsi import __main__ as cli
    from src.manager.mcq_rsi import splits
    from src.manager.mcq_rsi import benchmarks as registry
    pool, _ = make_pool(tmp_path)
    monkeypatch.setattr(cli, "_make_pool", lambda args, bench: pool)
    monkeypatch.setattr(splits, "read_manifest", lambda path: {"benchmark": "medqa"})
    monkeypatch.setattr(splits, "pool_rows", lambda manifest, name: make_rows(4) if name == "dev" else [])
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    resolved = []
    monkeypatch.setattr(E, "resolve_base", lambda m, r: resolved.append((m, r)) or str(snapshot))
    argv = ["evaluate", "--bench", "medqa", "--pool", "dev", "--checkpoint", str(checkpoint), "--out", str(tmp_path / "e")]
    assert cli.main(argv) == 0
    # The eval runs on the pinned snapshot of BASE_MODEL@BASE_REVISION, the base FA-GRPO loads.
    assert resolved[0] == (registry.BASE_MODEL, registry.BASE_REVISION)
    assert set(resolved[1:]) == {(str(snapshot), registry.BASE_REVISION)}
    signature = json.loads((tmp_path / "e" / "mcq_rsi_eval.json").read_text())["signature"]
    assert signature["base_model"] == str(snapshot) and signature["base_revision"] == registry.BASE_REVISION
    assert json.loads((tmp_path / "e" / "mcq_rsi_eval.json").read_text())["metrics"]["accuracy"] == 0.5
    scripted.model.script = {eid: [turns[-1]] for eid, turns in SCRIPT.items()}
    assert cli.main(argv[:-2] + ["--out", str(tmp_path / "f"), "--forced", "extractor,reasoner,verifier"]) == 0
    assert (tmp_path / "f" / "manager_forced_extractor_reasoner_verifier_report.json").exists()


def test_evaluate_pins_the_base_revision(tmp_path, scripted, checkpoint, pinned_base):
    """A hub id is replaced by its BASE_REVISION snapshot before stages loads it; both are in the signature."""
    from src.manager.mcq_rsi import benchmarks as registry
    pool, _ = make_pool(tmp_path)
    result = E.evaluate(checkpoint, make_rows(4), pool, tmp_path / "dev", bench="medqa")
    sig = result["signature"]
    assert pinned_base.calls[0] == (registry.BASE_MODEL, registry.BASE_REVISION)
    assert set(scripted.bases) == {str(pinned_base.snapshot)}  # pass 1 and the stages eval load the pinned base
    assert sig["base_model"] == str(pinned_base.snapshot) and sig["base_revision"] == registry.BASE_REVISION
    assert sig["base_identity"]["commit"] == registry.BASE_REVISION
    scripted.model.script = {eid: [turns[-1]] for eid, turns in SCRIPT.items()}
    forced = E.evaluate_forced(checkpoint, make_rows(4), pool, tmp_path / "forced", ["verifier"], bench="medqa")
    assert forced["signature"]["base_model"] == str(pinned_base.snapshot)
    assert forced["signature"]["base_revision"] == registry.BASE_REVISION
    assert scripted.bases[-1] == str(pinned_base.snapshot)
    # '' / None opts out explicitly (recorded); a different revision is a different eval.
    unpinned = E._signature("tools", checkpoint, make_rows(4), pool, "other/model", {}, None)
    assert unpinned["base_revision"] is None


def test_eval_code_identity_covers_the_eval_loop():
    from src.manager.mcq_rsi import collect
    names = {str(f) for src in collect.SOURCES for f in ([src] if src.is_file() else src.rglob("*.py"))}
    for f in ("pipeline/stages.py", "manager/prompt.py", "mcq_rsi/protocol.py", "mcq_rsi/prompts.py",
              "mcq_rsi/evaluate.py", "mcq_rsi/advisors.py"):
        assert any(n.endswith(f) for n in names), f
    assert E.code_identity()["source_sha256"] == collect.source_sha256()
