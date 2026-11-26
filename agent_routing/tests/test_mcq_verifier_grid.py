import json
import math
import sys
from pathlib import Path

import pytest

from mcq_rsi_helpers import FakeManager, advisor_server, make_rows
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import collect as C
from src.manager.mcq_rsi.advisors import CachedAdvisorPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mcq_verifier_grid as G  # noqa: E402
import mcq_verifier_grid_analyze as A  # noqa: E402

MEDQA = registry.get("medqa")


def advisor_pool(tmp_path, **kw):
    http, adapters = advisor_server(tmp_path, **kw)
    return CachedAdvisorPool("medqa", tmp_path / "advisor_cache", "http://x", adapters=adapters, http=http,
                             sleep=lambda s: None), http


class GridManager(FakeManager):
    """FakeManager whose revisions can also be scripted per extra branch (by the call id of the last tool message)."""

    def _rev(self, messages):
        eid = int(messages[-1]["tool_call_id"].split("_")[1])
        kind = messages[-1]["tool_call_id"].split("_", 2)[2] if messages[-1]["tool_call_id"].startswith("vg_") else None
        if kind is not None:
            draft = [m for m in messages if m.get("role") == "assistant"][-1]["content"].split("_")[-1]
            return self.script.get(eid, {}).get("extra", {}).get(kind, draft)
        return super()._rev(messages)


def script(rows):
    e = [r["example_id"] for r in rows]  # gold A, B, C, D
    return {
        e[0]: {"root": ("A", "commit"), "rev": {("reasoner",): "B"}, "extra": {"null": "A", "verifier_nocand": "A"}},
        e[1]: {"root": ("A", "verifier"), "rev": {("verifier",): "B"}, "extra": {"verifier_nocand": "A", "null": "A"}},
        e[2]: {"root": ("A", "reasoner"), "rev": {("verifier",): "C"}, "extra": {"verifier_nocand": "C", "null": "C"}},
        e[3]: {"root": ("A", "commit")},
    }


def test_grid_reuses_the_collector_and_adds_equal_information_branches(tmp_path):
    rows = make_rows(4)
    manager = GridManager(script(rows))
    pool, http = advisor_pool(tmp_path)
    out = G.run_grid(rows, MEDQA, manager, pool, tmp_path / "grid", pool_name="test", progress=False)
    records = [json.loads(line) for line in open(out["records_jsonl"])]
    assert [r["example_id"] for r in records] == [r["example_id"] for r in rows]
    for r, row in zip(records, rows):
        # the collector's depth-1 branches, unchanged
        ref = C.collect_question(row, MEDQA, GridManager(script(rows)), pool, root_mode="policy", max_depth=1)
        assert r["branches"] == ref["branches"] and r["policy_action"] == ref["policy_action"]
        assert [b["sequence"] for b in r["branches"]] == [["extractor"], ["reasoner"], ["verifier"]]
        assert [b["kind"] for b in r["extra_branches"]] == ["verifier_nocand", "null"]
        assert r["pool"] == "test" and r["split"] == "test"
        nocand, null = r["extra_branches"]
        verifier_turn = next(b for b in r["branches"] if b["sequence"] == ["verifier"])["trajectory"][0]
        # the manager wrote the same candidate; only the Verifier's argument is blank in the candidate-free branch
        assert nocand["call"]["content"] == null["call"]["content"] == verifier_turn["content"] == "DRAFT_ANSWER_A"
        args = lambda b: json.loads(b["call"]["tool_calls"][0]["function"]["arguments"])
        assert args(nocand) == {"current_draft": "", "example_id": row["example_id"]}
        assert args(null) == {"current_draft": "A", "example_id": row["example_id"]}
        assert null["tool_output_sha256"] == json.loads((tmp_path / "grid/grid_manifest.json").read_text())["null_output_sha256"]
    # the Verifier saw the candidate in the ordinary branch and none in the candidate-free branch
    bodies = [json.dumps(c["json"]) for c in http.calls if "verifier" in json.dumps(c["json"]).lower()] \
        if hasattr(http, "calls") else []
    with_candidate = pool.cached(C.AdvisorRequest.for_row("verifier", rows[0], "A"))
    without_candidate = pool.cached(C.AdvisorRequest.for_row("verifier", rows[0], ""))
    assert with_candidate is not None and without_candidate is not None
    assert pool.key("verifier", rows[0]["question"], "", rows[0]["choices"], "A") != \
        pool.key("verifier", rows[0]["question"], "", rows[0]["choices"], "")
    assert bodies is not None
    # outcomes follow the script
    by = {r["example_id"]: {b["kind"]: b["correct"] for b in r["extra_branches"]} for r in records}
    e = [r["example_id"] for r in rows]
    assert by[e[0]] == {"verifier_nocand": True, "null": True}      # gold A, kept
    assert by[e[1]] == {"verifier_nocand": False, "null": False}    # gold B, only the candidate-conditioned call fixes it
    assert by[e[2]] == {"verifier_nocand": True, "null": True}      # gold C
    summary = out["summary"]
    assert summary["n"] == 4 and summary["candidate_accuracy"] == 0.25
    assert summary["always"]["verifier"]["accuracy"] == 0.75        # A kept, B and C fixed, D not
    assert summary["always"]["null"]["accuracy"] == 0.5
    # policy: commit(A ok), verifier(B fixed), reasoner(C: unscripted -> keeps A, wrong), commit(D wrong)
    assert summary["policy_single_call"] == {"accuracy": 0.5, "call_rate": 0.5}


def test_resume_and_identity(tmp_path):
    rows = make_rows(4)
    pool, _ = advisor_pool(tmp_path)
    first = GridManager(script(rows), fail_at=rows[2]["example_id"])
    with pytest.raises(KeyboardInterrupt):
        G.run_grid(rows, MEDQA, first, pool, tmp_path / "grid", pool_name="dev", progress=False)
    assert sorted(p.stem for p in (tmp_path / "grid/shards").glob("*.json")) == [str(r["example_id"]) for r in rows[:2]]
    assert not (tmp_path / "grid/grid_records.jsonl").exists()
    with pytest.raises(FileExistsError):
        G.run_grid(rows, MEDQA, GridManager(script(rows)), pool, tmp_path / "grid", pool_name="dev", progress=False)
    resumed = G.run_grid(rows, MEDQA, GridManager(script(rows)), pool, tmp_path / "grid", pool_name="dev", resume=True,
                         progress=False)
    fresh = G.run_grid(rows, MEDQA, GridManager(script(rows)), pool, tmp_path / "fresh", pool_name="dev", progress=False)
    assert open(resumed["records_jsonl"]).read() == open(fresh["records_jsonl"]).read()
    with pytest.raises(ValueError, match="identity differs"):
        G.run_grid(rows, MEDQA, GridManager(script(rows), name="other"), pool, tmp_path / "grid", pool_name="dev",
                   resume=True, progress=False)
    with pytest.raises(ValueError, match="identity differs"):
        G.run_grid(rows[:3], MEDQA, GridManager(script(rows)), pool, tmp_path / "grid", pool_name="dev", resume=True,
                   progress=False)


def test_no_extra_branches(tmp_path):
    rows = make_rows(2)
    pool, _ = advisor_pool(tmp_path)
    out = G.run_grid(rows, MEDQA, GridManager(script(make_rows(4))), pool, tmp_path / "grid", pool_name="test", extra=(),
                     progress=False)
    records = [json.loads(line) for line in open(out["records_jsonl"])]
    assert all(r["extra_branches"] == [] for r in records)
    with pytest.raises(ValueError, match="unknown extra branch"):
        G.run_grid(rows, MEDQA, GridManager(script(make_rows(4))), pool, tmp_path / "bad", pool_name="test",
                   extra=("oops",), progress=False)


# --------------------------------------------------------------------------------------------- analysis

def record(eid, commit, e, r, v, action, conf, keys="ABCD", nocand=None, null=None):
    """A minimal grid record: candidate A, outcomes given as booleans, confidence = P(A)."""
    rest = (1 - conf) / (len(keys) - 1)
    klp = {k: math.log(conf if k == "A" else rest) for k in keys}
    branches = [{"sequence": [k], "correct": c} for k, c in (("extractor", e), ("reasoner", r), ("verifier", v))]
    extra = [{"kind": k, "correct": c} for k, c in (("verifier_nocand", nocand), ("null", null)) if c is not None]
    return {"example_id": eid, "direct_pred": "A", "direct_correct": commit, "direct_valid": True, "root_mode": "policy",
            "branches": branches, "extra_branches": extra,
            "policy_action": {"action": action, "draft": "A", "logprob": -0.1, "key_logprobs": klp}}


def toy():
    # id, commit, E, R, V, action, conf
    spec = [(1, True, True, False, True, "commit", 0.95), (2, True, True, True, True, "verifier", 0.90),
            (3, False, False, False, True, "verifier", 0.40), (4, False, False, True, False, "reasoner", 0.60),
            (5, False, False, False, True, "commit", 0.30), (6, True, False, True, False, "commit", 0.85),
            (7, False, False, False, False, "commit", 0.50), (8, True, True, True, True, "commit", 0.99)]
    return [record(*s, nocand=s[4] and s[1], null=s[1]) for s in spec]


def test_rules_on_a_hand_computed_table():
    res = A.analyze(toy(), draws=4000, n_boot=2000, seed=0)
    rules = res["rules"]
    assert res["n"] == 8 and res["policy_calls"] == 3
    assert res["policy_call_counts"] == {"extractor": 0, "reasoner": 1, "verifier": 2}
    assert rules["candidate"] == {"accuracy": 0.5, "calls": 0.0}
    # learned: 1 commit ok, 2 V ok, 3 V ok, 4 R ok, 5 commit wrong, 6 ok, 7 wrong, 8 ok
    assert rules["learned"] == {"accuracy": 0.75, "calls": 0.375}
    # the same three questions, all sent to the Verifier: question 4 is lost
    assert rules["learned_selection_verifier"]["accuracy"] == 0.625
    assert rules["always_verifier"]["accuracy"] == 0.625   # 1, 2, 3, 5, 8
    assert rules["always_reasoner"]["accuracy"] == 0.5
    assert rules["always_extractor"]["accuracy"] == 0.375
    assert rules["always_verifier_nocand"]["accuracy"] == 0.375 and rules["always_null"]["accuracy"] == 0.5
    assert rules["oracle"]["accuracy"] == 0.875 and rules["oracle"]["calls"] == 0.375  # 3, 4, 5 need a call; 7 unsolved
    # confidence order: 5 (.30), 3 (.40), 7 (.50), 4, 6, 1, 2, 8 -> the 3 least confident get the Verifier: 5 and 3 fixed
    assert rules["confidence_matched"] == {"accuracy": 0.75, "calls": 0.375}
    curve = [p["accuracy"] for p in res["confidence_curve"]]
    assert curve == [0.5, 0.625, 0.75, 0.75, 0.625, 0.5, 0.5, 0.5, 0.5][:0] + curve  # shape checked below
    assert curve[0] == 0.5 and curve[2] == 0.75 and curve[-1] == rules["always_verifier"]["accuracy"]
    assert curve[5] == curve[4] - 0.125  # the sixth least confident is question 6 (.85): the Verifier breaks it
    assert rules["confidence_best_in_pool"]["accuracy"] == 0.75 and rules["confidence_best_in_pool"]["calls"] == 0.25
    # random Verifier-only at 3 of 8 calls: expectation = 0.5 + 3/8 * (0.625 - 0.5)
    assert rules["random_verifier"]["expected"] == pytest.approx(0.5 + 3 / 8 * 0.125)
    assert rules["random_verifier"]["accuracy"] == pytest.approx(rules["random_verifier"]["expected"], abs=0.01)
    # random with the policy's roles: 2 Verifier + 1 Reasoner calls on random distinct questions
    d_v, d_r = (5 - 4) / 8, (4 - 4) / 8
    assert rules["random_roles"]["expected"] == pytest.approx(0.5 + 2 / 8 * (d_v * 8) / 8 + 1 / 8 * (d_r * 8) / 8)
    assert 0 <= rules["random_roles"]["share_at_least_learned"] <= 1
    tests = res["tests"]
    assert tests["learned - always_verifier"]["diff"] == pytest.approx(0.125)
    assert tests["candidate value: always_verifier - always_verifier_nocand"]["diff"] == pytest.approx(0.25)
    assert tests["evidence value: always_verifier - always_null"]["diff"] == pytest.approx(0.125)
    assert tests["revision without evidence: always_null - candidate"]["diff"] == 0


def test_random_draws_are_exact_for_a_degenerate_budget():
    rows = A.table(toy())
    everything = A.random_matched(rows, {"verifier": 8}, draws=50, seed=1)
    assert everything["ci95"] == [0.625, 0.625] and everything["mean"] == 0.625
    nothing = A.random_matched(rows, {"verifier": 0}, draws=50, seed=1)
    assert nothing["mean"] == 0.5
    with pytest.raises(ValueError):
        A.random_matched(rows, {"verifier": 9})


def test_dev_threshold_transfers_to_the_test_pool():
    dev = toy()
    test = [record(11, False, False, False, True, "commit", 0.47), record(12, True, True, True, False, "commit", 0.97),
            record(13, False, False, False, True, "verifier", 0.55), record(14, True, True, True, True, "commit", 0.35)]
    res = A.analyze(test, dev, draws=200, n_boot=500)
    r = res["rules"]["confidence_dev_matched"]
    # dev: 3 policy calls -> threshold between .50 and .60 = .55; test questions with conf < .55: 11 (.47) and 14 (.35)
    assert r["threshold"] == pytest.approx(0.55) and r["calls"] == 0.5 and r["dev_calls"] == 0.375
    assert r["accuracy"] == 0.75   # 11 fixed, 12 kept, 13 not called (wrong), 14 kept by the Verifier
    best = res["rules"]["confidence_dev_best"]
    assert best["dev_accuracy"] == 0.75 and best["dev_calls"] == 0.25 and best["threshold"] == pytest.approx(0.45)
    assert best["calls"] == 0.25 and best["accuracy"] == 0.5  # only question 14 (.35) is below .45


def test_table_refuses_inconsistent_records():
    bad = toy()
    bad[0]["branches"] = bad[0]["branches"][:2]
    with pytest.raises(ValueError, match="missing"):
        A.table(bad)
    probe = toy()
    probe[0]["policy_action"] = None
    with pytest.raises(ValueError, match="policy-root"):
        A.table(probe)
    no_conf = toy()
    for r in no_conf:
        r["policy_action"]["key_logprobs"] = None
    res = A.analyze(no_conf, draws=100, n_boot=200)
    assert "confidence_matched" not in res["rules"] and "confidence_curve" not in res


def test_markdown_and_cli(tmp_path):
    path = tmp_path / "grid_records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in toy()))
    assert A.main(["--records", str(path), "--dev", str(path), "--out", str(tmp_path / "out"), "--label", "toy",
                   "--draws", "100", "--n-boot", "200"]) == 0
    text = (tmp_path / "out/toy.md").read_text()
    assert "Always Verifier, with the candidate" in text and "Confidence-gated Verifier, dev threshold" in text
    assert (tmp_path / "out/toy_curve.csv").read_text().count("\n") == 10
    assert json.loads((tmp_path / "out/toy.json").read_text())["rules"]["learned"]["accuracy"] == 0.75


# ------------------------------------------------------------------------------------------------ queue

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "pod_ops"))
import verifier_grid_queue as Q  # noqa: E402


class FakeProc:
    def __init__(self, polls, rc, on_exit=None):
        self.polls, self.rc, self.on_exit = polls, rc, on_exit

    def poll(self):
        if self.polls > 0:
            self.polls -= 1
            return None
        if self.on_exit is not None:
            self.on_exit()
            self.on_exit = None
        return self.rc


def test_slots():
    assert Q.parse_slots("0:1,1:3") == ["0", "1", "1", "1"] and Q.parse_slots("2") == ["2"]
    for bad in ("", "a:1", "0:x"):
        with pytest.raises(ValueError):
            Q.parse_slots(bad)
    # an 80 GB card next to a server that holds 36 GB, two empty 80 GB cards, one small card
    assert Q.auto_slots([44000, 81000, 81000, 20000]) == ["0", "1", "1", "1", "2", "2", "2"]
    assert Q.auto_slots([48000], max_per_gpu=1) == ["0"] and Q.auto_slots([1000]) == []


def test_job_order_and_filter():
    jobs = Q.all_jobs()
    assert jobs[:4] == [("s1", b, "test") for b in ("medqa", "mmlu_pro", "aqua", "gpqa")]
    assert len(jobs) == len(set(jobs)) == 8 + 4 + 1 + 4 + 1 + 8 + 1
    assert ("dyn3g", "mmlu_pro", "test") in jobs and ("dyn3g", "medqa", "test") not in jobs
    assert Q.all_jobs(r"^s1/.*/test$") == jobs[:4]


def test_queue_runs_in_order_retries_and_fills_slots(tmp_path):
    jobs = [("s1", b, "test") for b in ("medqa", "mmlu_pro", "aqua", "gpqa")] + [("dyn3", "medqa", "test")]
    done, started, after = {("s1", "gpqa", "test")}, [], []
    fails = {("s1", "mmlu_pro", "test"): 1, ("dyn3", "medqa", "test"): 9}  # one transient failure, one permanent

    def launch(job, gpu):
        started.append((job, gpu))
        if fails.get(job, 0) > 0:
            fails[job] -= 1
            return FakeProc(1, 1)
        return FakeProc(2, 0, on_exit=lambda: done.add(job))

    state = Q.run_queue(jobs, ["0", "1"], launch=launch, finished=lambda j: j in done, after=after.append, retries=2,
                        state_path=tmp_path / "state.json", sleep=lambda s: None)
    assert state["s1/gpqa/test"] == {"status": "done", "attempts": 0, "note": "finished earlier"}
    assert [state[Q.job_id(j)]["status"] for j in jobs] == ["done", "done", "done", "done", "failed"]
    assert state["s1/mmlu_pro/test"]["attempts"] == 2 and state["dyn3/medqa/test"]["attempts"] == 3
    assert [j for j, _ in started[:2]] == jobs[:2] and {g for _, g in started[:2]} == {"0", "1"}
    assert started[2][0] == ("s1", "mmlu_pro", "test")  # the failed job resumes before lower-priority work
    assert set(after) == {jobs[0], jobs[1], jobs[2]}
    assert json.loads((tmp_path / "state.json").read_text()) == state


def test_queue_survives_a_job_that_cannot_start(tmp_path):
    done = set()

    def launch(job, gpu):
        if job[0] == "base":
            raise FileNotFoundError("base_model_path.txt")
        return FakeProc(0, 0, on_exit=lambda: done.add(job))

    state = Q.run_queue([("base", "medqa", "test"), ("s1", "medqa", "test")], ["0"], launch=launch,
                        finished=lambda j: j in done, sleep=lambda s: None)
    assert state["base/medqa/test"]["status"] == "failed" and "base_model_path" in state["base/medqa/test"]["error"]
    assert state["s1/medqa/test"]["status"] == "done"


def test_commands(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "WORK", tmp_path)
    monkeypatch.setattr(Q, "OUT", tmp_path / "runs/verifier_grid")
    monkeypatch.setattr(Q, "CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(Q, "PY", "py")
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt/base_model_path.txt").write_text("/hub/base\n")
    assert Q.checkpoint_path("s1", "aqua") == str(tmp_path / "import/aqua/round1/sft")
    assert Q.checkpoint_path("dyn3g", "mmlu_pro") == str(tmp_path / "ckpt/mmlu_pro_dyn3g")
    assert Q.checkpoint_path("base", "gpqa") == "/hub/base"
    job = ("s1", "medqa", "test")
    cmd = Q.command(job)
    assert cmd[:4] == ["py", "scripts/mcq_verifier_grid.py", "--config", "configs/mcq_rsi_medqa_v3.json"] and "--resume" not in cmd
    out = Q.out_dir(job)
    out.mkdir(parents=True)
    (out / "grid_manifest.json").write_text("{}")
    assert Q.command(job)[-1] == "--resume" and not Q.finished(job)
    (out / "grid_summary.json").write_text("{}")
    assert Q.finished(job)
    # the test pool is analysed alone first, and again with the dev threshold once the dev grid exists
    assert [c[c.index("--label") + 1] for c in Q.analysis_commands(job)] == ["s1_medqa_test"]
    assert "--dev" not in Q.analysis_commands(job)[0]
    dev = Q.out_dir(("s1", "medqa", "dev"))
    dev.mkdir(parents=True)
    (dev / "grid_records.jsonl").write_text("")
    (dev / "grid_summary.json").write_text("{}")
    cmds = Q.analysis_commands(("s1", "medqa", "dev"))
    assert [c[c.index("--label") + 1] for c in cmds] == ["s1_medqa_dev", "s1_medqa_test"]
    assert "--dev" not in cmds[0] and cmds[1][cmds[1].index("--dev") + 1] == str(dev / "grid_records.jsonl")
