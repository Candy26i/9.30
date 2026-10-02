import json
import random

import pytest

from mcq_rsi_helpers import FakeManager, advisor_server, make_rows, read_fixture, tiny_model, tiny_tokenizer
from src.manager.marginal_value import _make_sft_rows, summarize_counterfactuals
from src.manager.mcq_rsi import __main__ as cli
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import collect as C
from src.manager.mcq_rsi import protocol, splits
from src.manager.mcq_rsi.advisors import AdvisorError, CachedAdvisorPool
from src.pipeline import stages

MEDQA = registry.get("medqa")
PAPER_RECORD_KEYS = ["example_id", "question_hash", "benchmark_name", "ground_truth", "direct_pred", "direct_valid",
                     "direct_correct", "direct_text", "preferred_sequence", "base_messages", "branches"]
PAPER_BRANCH_KEYS = ["example_id", "question_hash", "sequence", "depth", "initial_draft", "drafts", "final_pred",
                     "valid", "correct", "probe_text", "trajectory"]


def advisor_pool(tmp_path, **kw):
    http, adapters = advisor_server(tmp_path, **kw)
    return CachedAdvisorPool("medqa", tmp_path / "advisor_cache", "http://x", adapters=adapters, http=http,
                             sleep=lambda s: None), http


def scripted(rows):
    """make_rows(4) (gold A, B, C, D): correct root / depth-1 tie / depth-2 rescue / unsolved."""
    e = [row["example_id"] for row in rows]
    return {
        e[0]: {"root": ("A", "commit"), "rev": {("reasoner",): "B"}},
        e[1]: {"root": ("A", "verifier"), "rev": {("extractor",): "B", ("verifier",): "B"}},
        e[2]: {"root": ("A", "commit"), "rev": {("reasoner", "verifier"): "C"}},
        e[3]: {"root": ("A", "reasoner")},
    }


def test_stop_rules_tie_break_and_schema(tmp_path):
    rows = make_rows(4)
    manager = FakeManager(scripted(rows))
    pool, http = advisor_pool(tmp_path)
    out = C.collect(rows, MEDQA, manager, pool, tmp_path / "run", pool_name="collect_r2", round_index=2)
    records = [json.loads(line) for line in open(out["records_jsonl"])]
    seqs = lambda r: [b["sequence"] for b in r["branches"]]
    depth1 = [["extractor"], ["reasoner"], ["verifier"]]
    depth2 = [["extractor", "reasoner"], ["extractor", "verifier"], ["reasoner", "extractor"],
              ["reasoner", "verifier"], ["verifier", "extractor"], ["verifier", "reasoner"]]
    correct, tie, deep, unsolved = records
    assert seqs(correct) == depth1 and correct["preferred_sequence"] == []  # correct root: depth 1 only
    assert not correct["branches"][1]["correct"]  # the reasoner corrupts it (corruption statistic)
    assert seqs(tie) == depth1 and tie["preferred_sequence"] in (["extractor"], ["verifier"])
    expected = random.Random(42 + tie["example_id"]).choice([("extractor",), ("verifier",)])
    assert tuple(tie["preferred_sequence"]) == expected
    assert seqs(deep) == depth1 + depth2 and deep["preferred_sequence"] == ["reasoner", "verifier"]
    assert seqs(unsolved) == depth1 + depth2 and unsolved["preferred_sequence"] is None
    # Paper record schema, then the extra fields.
    for r in records:
        assert list(r)[:len(PAPER_RECORD_KEYS)] == PAPER_RECORD_KEYS
        assert set(r) - set(PAPER_RECORD_KEYS) == {"split", "root_mode", "policy_action", "unconstrained_argmax",
                                                   "pool", "round"}
        assert r["split"] == "train" and r["pool"] == "collect_r2" and r["round"] == 2
        assert all(list(b)[:len(PAPER_BRANCH_KEYS)] == PAPER_BRANCH_KEYS for b in r["branches"])
    assert tie["policy_action"]["action"] == "verifier" and unsolved["policy_action"]["action"] == "reasoner"
    assert deep["unconstrained_argmax"] == {"checked": 6 + 9 * 9, "mismatch": 0, "would_call": 0}
    # Depth-2 states continue from the revised draft; the Verifier audits the current draft.
    branch = next(b for b in deep["branches"] if b["sequence"] == ["reasoner", "verifier"])
    assert branch["drafts"] == ["A", "A", "C"] and branch["probe_text"] == "DRAFT_ANSWER_C\nANSWER_C"
    call = branch["trajectory"][1]
    assert call["content"] == "DRAFT_ANSWER_A" and call["tool_calls"][0]["id"] == f"mv_{deep['example_id']}_reasoner_verifier"
    assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"current_draft": "A"}
    verifier_bodies = [p for p in http.posts if p["model"] == "medqa_verifier"]
    assert all("CANDIDATE ANSWER TO AUDIT" in p["messages"][-1]["content"] for p in verifier_bodies)
    # Revisions of one layer are batched: 3 per depth-1 layer, 6 per depth-2 layer.
    assert manager.calls["revise"] == [3, 3, 3, 6, 3, 6]
    # Accepted unchanged by the paper helpers.
    report = summarize_counterfactuals(records)
    assert report["direct_accuracy"] == 0.25 and report["oracle_accuracy"] == 0.75 and report["n_unsolved"] == 1
    assert out["report"]["policy_call_rate"] == 0.5 and out["report"]["n_branches"] == 3 + 3 + 9 + 9
    sft = [row for r in records for row in _make_sft_rows(r)]
    assert [row["decision_type"] for row in sft] == ["commit", "call", "commit_after_call",
                                                     "call", "call", "commit_after_call"]
    assert sft[-1]["prompt"][-1]["role"] == "tool" and sft[-1]["response"][0]["content"] == "DRAFT_ANSWER_C\nANSWER_C"


def test_stop_rule_edge_cases_and_revision_semantics(tmp_path):
    """make_rows(3) (gold A, B, C)."""
    rows = make_rows(3)
    e = [row["example_id"] for row in rows]
    script = {
        # Correct root, every depth-1 revision corrupts it: still no depth 2, label commit.
        e[0]: {"root": ("A", "commit"), "rev": {(k,): "D" for k in ("extractor", "reasoner", "verifier")}},
        # The reasoner moves the draft to D; the depth-2 Verifier must audit D, not the root draft.
        e[1]: {"root": ("A", "commit"), "rev": {("reasoner",): "D", ("reasoner", "verifier"): "B"}},
        # DRAFT_ANSWER_A\nANSWER_C: the outcome is the ANSWER key (paper + eval), not the draft.
        e[2]: {"root": ("A", "commit"), "rev": {("extractor",): ("A", "C")}},
    }
    manager = FakeManager(script)
    pool, http = advisor_pool(tmp_path)
    out = C.collect(rows, MEDQA, manager, pool, tmp_path / "run", pool_name="collect_r2", round_index=2)
    corrupted, moved, answer = [json.loads(line) for line in open(out["records_jsonl"])]
    assert len(corrupted["branches"]) == 3 and not any(b["correct"] for b in corrupted["branches"])
    assert corrupted["preferred_sequence"] == []
    branch = next(b for b in moved["branches"] if b["sequence"] == ["reasoner", "verifier"])
    assert branch["drafts"] == ["A", "D", "B"] and moved["preferred_sequence"] == ["reasoner", "verifier"]
    call = branch["trajectory"][1]
    assert call["content"] == "DRAFT_ANSWER_D"
    assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {"current_draft": "D"}
    audited = {c for c in "ABCD" if pool._path(pool.key("verifier", rows[1]["question"], "", rows[1]["choices"], c)).exists()}
    assert audited == {"A", "D"}  # V(A) at depth 1, V(D) after the reasoner
    b = answer["branches"][0]
    assert b["sequence"] == ["extractor"] and b["final_pred"] == "C" and b["correct"] and b["drafts"] == ["A", "C"]
    assert b["probe_text"] == "DRAFT_ANSWER_A\nANSWER_C" and b["revision"]["draft"] == "A" and b["revision"]["would_call"]
    assert answer["preferred_sequence"] == ["extractor"]
    assert _make_sft_rows(answer)[-1]["response"][0]["content"] == "DRAFT_ANSWER_C\nANSWER_C"  # paper target
    assert out["report"]["revision_answer_draft_mismatch"] == 1
    assert out["report"]["unconstrained_argmax"]["would_call"] == 1


def test_policy_revisions_see_the_eval_call_turn(tmp_path):
    rows = make_rows(1)
    eid = rows[0]["example_id"]
    manager = FakeManager({eid: {"root": ("B", "commit")}})
    pool, _ = advisor_pool(tmp_path)
    record = C.collect_question(rows[0], MEDQA, manager, pool)
    for state in manager.calls["revise_states"]:
        call = state[-2]
        args = json.loads(call["tool_calls"][0]["function"]["arguments"])
        kind = call["tool_calls"][0]["function"]["name"][:-5]
        assert args == ({"current_draft": "B"} if kind == "verifier" else {}) | {"example_id": eid}
        assert call == protocol.eval_call_message(kind, "B", eid, call["tool_calls"][0]["id"])
    for b in record["branches"]:  # stored trajectories keep the paper call turn (SFT rows unchanged)
        args = json.loads(b["trajectory"][0]["tool_calls"][0]["function"]["arguments"])
        assert "example_id" not in args
    probe = FakeManager({eid: {"root": ("B", "commit")}})
    seen = []
    probe.probe = lambda messages, keys, p: (seen.append(messages), ("B", "DRAFT_ANSWER_B\nANSWER_B", True))[1]
    C.collect_question(rows[0], MEDQA, probe, pool, root_mode="probe")
    assert all("example_id" not in m["tool_calls"][0]["function"]["arguments"]
               for msgs in seen for m in msgs if m.get("tool_calls"))


class RecordingBackend(protocol.HFBackend):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rendered = []

    def prompt_ids(self, messages, tools):
        self.rendered.append((messages, tools))
        return super().prompt_ids(messages, tools)


def test_each_stage_renders_its_schema(tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(1)
    tok = tiny_tokenizer()
    backend = RecordingBackend(tiny_model(tok), tok)
    manager = protocol.Manager(backend)
    assert manager.tools is protocol.TOOLS_DEPLOY
    rows = make_rows(1)
    pool, _ = advisor_pool(tmp_path)
    C.collect_question(rows[0], MEDQA, manager, pool, max_depth=1)
    rendered = list(backend.rendered)  # snapshot: prompt_ids below appends to backend.rendered
    assert len(rendered) == 4 and all(tools is protocol.TOOLS_DEPLOY for _, tools in rendered)
    assert rendered[0][0] == protocol.manager_messages(MEDQA, rows[0])
    deploy_text = protocol.render(tok, rendered[0][0], protocol.TOOLS_DEPLOY)
    assert deploy_text == stages._render_manager_chat(tok, rendered[0][0], stages._manager_tool_schemas("environment"))
    assert deploy_text != protocol.render(tok, rendered[0][0], protocol.TOOLS_SFT)
    for messages, _ in rendered:
        assert protocol.HFBackend.prompt_ids(backend, messages, protocol.TOOLS_DEPLOY) == \
            list(tok(protocol.render(tok, messages, protocol.TOOLS_DEPLOY))["input_ids"])
    used = []
    monkeypatch.setattr(protocol, "_generate_answer", lambda tok_, model, messages, tools, keys, probe, *a:
                        (used.append(tools), ("A", "DRAFT_ANSWER_A\nANSWER_A", True))[1])
    C.collect_question(rows[0], MEDQA, manager, pool, root_mode="probe", max_depth=1)
    assert used and all(tools is protocol.TOOLS_SFT for tools in used)


def test_failed_lookahead_stops_before_its_root(tmp_path):
    rows = make_rows(3)
    bad = rows[1]["question"]
    pool, _ = advisor_pool(tmp_path, fail_when=lambda body: bad in body["messages"][-1]["content"]
                           and body["model"] == "medqa_extractor")
    manager = FakeManager(scripted(make_rows(4)))
    with pytest.raises(AdvisorError):
        C.collect(rows, MEDQA, manager, pool, tmp_path / "run", pool_name="collect_r2", round_index=2)
    assert manager.calls["root"] == [rows[0]["example_id"]]
    assert pool.stats["failed"] >= 1


def test_failure_aborts_lookahead_retries(tmp_path):
    """A failed root aborts the pool: a lookahead request of a later root that is failing in the background makes
    no further attempts (it would otherwise retry ``retries`` times), and the next collect clears the abort."""
    import threading
    import time
    rows = make_rows(3)
    late = rows[2]["question"]
    release, posted = threading.Event(), threading.Event()
    is_late = lambda body: late in body["messages"][-1]["content"] and body["model"] == "medqa_extractor"

    def hold(body):
        if is_late(body):
            posted.set()
            return release
        return None
    pool, http = advisor_pool(tmp_path, fail_when=is_late, hold=hold)
    pool.retries = 3

    class KilledOnRoot1(FakeManager):
        def root(self, messages, keys, kinds, example_id):
            if example_id == rows[1]["example_id"]:
                assert posted.wait(10)  # the late root's lookahead attempt is in flight
            return super().root(messages, keys, kinds, example_id)

    script = scripted(make_rows(4))
    with pytest.raises(KeyboardInterrupt):
        C.collect(rows, MEDQA, KilledOnRoot1(script, fail_at=rows[1]["example_id"]), pool, tmp_path / "run",
                  pool_name="collect_r2", round_index=2)
    release.set()
    deadline = time.monotonic() + 10
    while pool.stats["failed"] < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pool.stats["failed"] == 1 and pool.stats["retries"] == 0
    assert sum(is_late(body) for body in http.posts) == 1
    # A new collect starts with the abort cleared.
    http.fail_when = None
    out = C.collect(rows, MEDQA, FakeManager(script), pool, tmp_path / "run", pool_name="collect_r2", round_index=2,
                    resume=True)
    assert len(open(out["records_jsonl"]).readlines()) == 3


def test_orphan_shards_are_not_adopted(tmp_path):
    rows = make_rows(4)
    script = scripted(rows)
    pool, _ = advisor_pool(tmp_path)
    run = tmp_path / "run"
    C.collect(rows[:2], MEDQA, FakeManager(script), pool, run, pool_name="collect_r2", round_index=2, seed=1)
    (run / "collect_manifest.json").unlink()
    with pytest.raises(FileExistsError, match="without a collect_manifest"):
        C.collect(rows, MEDQA, FakeManager(script), pool, run, pool_name="collect_r2", round_index=2)
    assert not (run / "collect_manifest.json").exists()


def test_resume_validates_shards_not_just_file_names(tmp_path):
    rows = make_rows(4)
    script = scripted(rows)
    pool, _ = advisor_pool(tmp_path)
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        C.collect(rows, MEDQA, FakeManager(script, fail_at=rows[2]["example_id"]), pool, run,
                  pool_name="collect_r2", round_index=2)
    good = run / "shards" / f"{rows[0]['example_id']}.json"
    original = good.read_text()
    other = tmp_path / "other"  # a shard of another pool's run under the right file name
    C.collect(rows[:1], MEDQA, FakeManager(script), pool, other, pool_name="collect_r3", round_index=3)
    cases = {
        "pool": (other / "shards" / good.name).read_text(),
        "question_hash": original.replace(json.loads(original)["question_hash"], "0" * 64),
        "unreadable": original[:40],
    }
    for what, text in cases.items():
        good.write_text(text)
        with pytest.raises(ValueError, match=what if what != "unreadable" else "unreadable shard"):
            C.collect(rows, MEDQA, FakeManager(script), pool, run, pool_name="collect_r2", round_index=2, resume=True)
    good.write_text(original)
    stray = run / "shards" / "999.json"
    stray.write_text(original)
    with pytest.raises(ValueError, match="outside this pool"):
        C.collect(rows, MEDQA, FakeManager(script), pool, run, pool_name="collect_r2", round_index=2, resume=True)
    stray.unlink()
    resumed = FakeManager(script)
    C.collect(rows, MEDQA, resumed, pool, run, pool_name="collect_r2", round_index=2, resume=True)
    assert resumed.calls["root"] == [r["example_id"] for r in rows[2:]]


def test_revision_states_render_as_the_deployed_eval_history(tmp_path):
    """The manager sees each post-call state exactly as ``stages.run_eval_manager_tools`` builds it."""
    tok = tiny_tokenizer()
    rows = make_rows(1)
    row, eid = rows[0], rows[0]["example_id"]
    manager = FakeManager({eid: {"root": ("B", "commit"), "rev": {("reasoner",): "D"}}})
    pool, _ = advisor_pool(tmp_path)
    C.collect_question(row, MEDQA, manager, pool, max_depth=2)
    states = {tuple(c["function"]["name"][:-5] for m in st if m.get("tool_calls") for c in m["tool_calls"]): st
              for st in manager.calls["revise_states"]}

    def eval_history(calls):
        """stages.py: the parsed call's content and arguments + injected example_id, call ids eval_<eid>_<n>."""
        messages = protocol.manager_messages(MEDQA, row)
        for n, (kind, draft) in enumerate(calls):
            args = {"current_draft": draft} if kind == "verifier" else {}
            args["example_id"] = eid
            messages.append(stages._tool_call_message(f"{kind}_tool", args, f"eval_{eid}_{n}", f"DRAFT_ANSWER_{draft}"))
            messages.append({"role": "tool", "tool_call_id": f"eval_{eid}_{n}", "name": f"{kind}_tool",
                             "content": pool.call(kind, eid, row["question"], "", row["choices"],
                                                  candidate_answer=draft if kind == "verifier" else "")})
        return messages

    render = lambda m: stages._render_manager_chat(tok, m, stages._manager_tool_schemas("environment"))
    assert protocol.render(tok, states[("verifier",)], protocol.TOOLS_DEPLOY) == render(eval_history([("verifier", "B")]))
    assert protocol.render(tok, states[("reasoner", "verifier")], protocol.TOOLS_DEPLOY) == \
        render(eval_history([("reasoner", "B"), ("verifier", "D")]))
    # Without the injected example_id (the paper SFT call turn) the prompt would differ from deployment.
    paper = [m if not m.get("tool_calls") else protocol.call_message(
        m["tool_calls"][0]["function"]["name"][:-5], "B", eid, "x") for m in states[("verifier",)]]
    assert protocol.render(tok, paper, protocol.TOOLS_DEPLOY) != render(eval_history([("verifier", "B")]))


def test_kill_and_resume_is_byte_identical(tmp_path):
    rows = make_rows(4)
    script = scripted(rows)
    pool, _ = advisor_pool(tmp_path)
    ref = tmp_path / "ref"
    C.collect(rows, MEDQA, FakeManager(script), pool, ref, pool_name="collect_r2", round_index=2)
    run = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        C.collect(rows, MEDQA, FakeManager(script, fail_at=rows[2]["example_id"]), pool, run,
                  pool_name="collect_r2", round_index=2)
    assert sorted(p.name for p in (run / "shards").iterdir()) == [f"{r['example_id']}.json" for r in rows[:2]]
    assert not list(run.rglob("*.part")) and not (run / "counterfactual_records.jsonl").exists()
    with pytest.raises(FileExistsError):
        C.collect(rows, MEDQA, FakeManager(script), pool, run, pool_name="collect_r2", round_index=2)
    with pytest.raises(ValueError, match="manager"):
        C.collect(rows, MEDQA, FakeManager(script, name="other"), pool, run, pool_name="collect_r2", round_index=2,
                  resume=True)
    resumed = FakeManager(script)
    C.collect(rows, MEDQA, resumed, pool, run, pool_name="collect_r2", round_index=2, resume=True)
    assert resumed.calls["root"] == [r["example_id"] for r in rows[2:]]
    for name in ("counterfactual_records.jsonl", "counterfactual_branches.jsonl", "marginal_value_report.json",
                 "collect_manifest.json"):
        assert (run / name).read_bytes() == (ref / name).read_bytes()
    for r in rows:
        assert (run / "shards" / f"{r['example_id']}.json").read_bytes() == \
               (ref / "shards" / f"{r['example_id']}.json").read_bytes()


def test_advisor_failure_stops_before_the_shard(tmp_path):
    rows = make_rows(3)
    bad = rows[1]["question"]
    pool, _ = advisor_pool(tmp_path, fail_when=lambda body: bad in body["messages"][-1]["content"]
                           and body["model"] == "medqa_verifier")
    with pytest.raises(AdvisorError):
        C.collect(rows, MEDQA, FakeManager(scripted(make_rows(4))), pool, tmp_path / "run",
                  pool_name="collect_r2", round_index=2)
    assert [p.name for p in (tmp_path / "run" / "shards").iterdir()] == [f"{rows[0]['example_id']}.json"]
    assert not any("error" in p.read_text() for p in (tmp_path / "advisor_cache").rglob("*.json"))


def test_end_to_end_tiny_hf_model(tmp_path):
    import torch
    torch.set_num_threads(1)
    tok = tiny_tokenizer()
    manager = protocol.Manager(protocol.HFBackend(tiny_model(tok), tok, batch_size=2, identity={"checkpoint": "tiny"}))
    rows = make_rows(2)
    pool, _ = advisor_pool(tmp_path)
    out = C.collect(rows, MEDQA, manager, pool, tmp_path / "run", pool_name="collect_r2", round_index=2)
    records = [json.loads(line) for line in open(out["records_jsonl"])]
    for r in records:
        assert r["direct_valid"] and r["direct_pred"] in "ABCD"
        assert r["policy_action"]["action"] in ("commit", "extractor", "reasoner", "verifier")
        assert r["policy_action"]["draft"] == r["direct_pred"]
        assert r["unconstrained_argmax"]["checked"] > 0
        assert len(r["branches"]) in (3, 9) and all(b["valid"] for b in r["branches"])
        assert all(b["probe_text"] == f"DRAFT_ANSWER_{b['final_pred']}\nANSWER_{b['final_pred']}" for b in r["branches"])
        _make_sft_rows(r)
    summarize_counterfactuals(records)
    # Probe mode (paper generation) runs on the same model.
    probe = protocol.Manager(protocol.HFBackend(manager.backend.model, tok), probe_max_new_tokens=4)
    rec = C.collect_question(rows[0], MEDQA, probe, pool, root_mode="probe")
    assert rec["policy_action"] is None and rec["unconstrained_argmax"] is None


class ReplayManager:
    """Answers the paper probes with what a recorded paper-era tree says the model generated."""

    def __init__(self, records):
        self.records = {r["example_id"]: r for r in records}

    def identity(self):
        return {"replay": "paper"}

    def probe(self, messages, keys, probe):
        from mcq_rsi_helpers import example_id_of, sequence_of
        rec, seq = self.records[example_id_of(messages)], list(sequence_of(messages))
        if not seq:
            assert probe.startswith("Training-time counterfactual probe: do not call a tool")
            return rec["direct_pred"], rec["direct_text"], rec["direct_valid"]
        branch = next(b for b in rec["branches"] if b["sequence"] == seq)
        return branch["final_pred"], branch["probe_text"], branch["valid"]


def seeded_pool(tmp_path, rows, records):
    pool = CachedAdvisorPool("medqa", tmp_path / "advisor_cache")  # offline: every output must be seeded
    for row, rec in zip(rows, records):
        for branch in rec["branches"]:
            for event in branch["trajectory"]:
                kind = event["tool_event"]["name"][:-5]
                args = json.loads(event["tool_calls"][0]["function"]["arguments"])
                pool.put(kind, row["question"], row["context"], row["choices"], event["tool_event"]["content"],
                         candidate=args.get("current_draft", ""))
    return pool


def test_paper_parity_replay_records_and_eval():
    """Recorded paper-era MedQA trees (d2_400, seed 42) replay through the collector unchanged."""
    import tempfile
    from pathlib import Path
    fx = read_fixture("paper_medqa_replay.json")
    with tempfile.TemporaryDirectory() as tmp:
        pool = seeded_pool(Path(tmp), fx["rows"], fx["records"])
        for row, paper in zip(fx["rows"], fx["records"]):
            ours = C.collect_question(row, MEDQA, ReplayManager(fx["records"]), pool, root_mode="probe", max_depth=2)
            assert {k: ours[k] for k in PAPER_RECORD_KEYS} == paper
            assert list(ours)[:len(PAPER_RECORD_KEYS)] == list(paper)
            assert [list(b) for b in ours["branches"]] == [list(b) for b in paper["branches"]]
    labels = [r["preferred_sequence"] for r in fx["records"]]
    assert None in labels and [] in labels and {1, 2} <= {len(x) for x in labels if x}
    # Deployed S_1 eval trajectories lie in the grammar and give the record's policy_action shape.
    for row in fx["eval"]:
        turns = C.policy_turns(row, list("ABCD"))
        assert turns and None not in turns and turns[-1][1] is None and len(turns) == row["tool_calls"] + 1
        action = C.policy_action_from_eval(row, list("ABCD"))
        assert action["draft"] == row["initial_draft"]
        assert action["action"] == (row["tool_names_called"][0][:-5] if row["tool_names_called"] else "commit")


def test_cli_collect_tiny_checkpoint(tmp_path, monkeypatch):
    """The CLI path end to end on CPU: real LoRA checkpoint loading, fake advisor server."""
    from peft import LoraConfig, get_peft_model
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    model.save_pretrained(tmp_path / "base")
    get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"])).save_pretrained(tmp_path / "sft")
    tok.save_pretrained(tmp_path / "sft")
    rows = make_rows(2)
    pool, _ = advisor_pool(tmp_path)
    monkeypatch.setattr(cli, "_make_pool", lambda args, bench: pool)
    monkeypatch.setattr(splits, "read_manifest", lambda path: {"benchmark": "medqa"})
    monkeypatch.setattr(splits, "pool_rows", lambda manifest, name: rows if name == "collect_r2" else [])
    argv = ["collect", "--bench", "medqa", "--pool", "collect_r2", "--checkpoint", str(tmp_path / "sft"),
            "--base-model", str(tmp_path / "base"), "--out", str(tmp_path / "run"), "--advisor-url", "http://x"]
    assert cli.main(argv) == 0
    report = json.loads((tmp_path / "run" / "marginal_value_report.json").read_text())
    assert report["round"] == 2 and report["n_examples"] == 2 and report["max_depth"] == 2
    manifest = json.loads((tmp_path / "run" / "collect_manifest.json").read_text())
    assert len(manifest["manager"]["adapter_sha256"]) == 64 and manifest["advisors"]["bench"] == "medqa"
    assert manifest["manager"]["batch_size"] == 1 and manifest["manager"]["device_type"] == "cpu"
    assert manifest["harness"]["packages"]["torch"] and len(manifest["source_sha256"]) == 64
    assert {"flash-linear-attention", "causal-conv1d"} <= set(manifest["harness"]["packages"])
    assert manifest["manager"]["dtype"] == "torch.float32" and len(manifest["manager"]["chat_template_sha256"]) == 64
    with pytest.raises(FileExistsError):
        cli.main(argv)
    before = (tmp_path / "run" / "counterfactual_records.jsonl").read_bytes()
    assert cli.main(argv + ["--resume"]) == 0
    assert (tmp_path / "run" / "counterfactual_records.jsonl").read_bytes() == before
    with pytest.raises(ValueError, match="manager"):  # a different decoding setup never resumes
        cli.main(argv + ["--resume", "--batch-size", "2"])
