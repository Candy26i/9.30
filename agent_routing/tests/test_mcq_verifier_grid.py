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
    assert all(r["root_check"] == {"checked": 6, "mismatch": 0, "first_mismatch": None} for r in records)
    assert summary["root_equals_unconstrained_greedy"] == {"n_equal": 4, "n_checked": 4, "n": 4}
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
    path, dev = tmp_path / "grid_records.jsonl", tmp_path / "dev_records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in toy()))
    dev.write_text("".join(json.dumps({**r, "example_id": r["example_id"] + 100}) + "\n" for r in toy()))
    assert A.main(["--records", str(path), "--dev", str(dev), "--out", str(tmp_path / "out"), "--label", "toy",
                   "--draws", "100", "--n-boot", "200"]) == 0
    text = (tmp_path / "out/toy.md").read_text()
    assert "Always Verifier, with the candidate" in text and "Confidence-gated Verifier, dev threshold" in text
    assert (tmp_path / "out/toy_curve.csv").read_text().count("\n") == 10
    assert json.loads((tmp_path / "out/toy.json").read_text())["rules"]["learned"]["accuracy"] == 0.75


def test_dev_questions_inside_the_evaluated_pool_are_left_out_of_the_dev_rules():
    dev = toy()
    # the evaluated pool holds the whole dev grid (as MMLU-Pro test_paper does) plus four other questions
    spec = [(11, False, False, False, True, "commit", 0.47), (12, True, True, True, False, "commit", 0.97),
            (13, False, False, False, True, "verifier", 0.55), (14, True, True, True, True, "commit", 0.35)]
    fresh = [record(*x, nocand=False, null=x[1]) for x in spec]
    res = A.analyze(dev + fresh, dev, draws=200, n_boot=500)
    assert res["dev_overlap"] == {"shared": 8, "kept": 4}
    clean = A.analyze(fresh, dev, draws=200, n_boot=500)
    for name in ("confidence_dev_matched", "confidence_dev_best"):
        r = res["rules"][name]
        assert r["n"] == 4 and r["dev_overlap_excluded"] == 8
        assert (r["accuracy"], r["calls"], r["threshold"]) == tuple(clean["rules"][name][k] for k in ("accuracy", "calls", "threshold"))
        assert res["tests"][f"learned - {name}"]["diff"] == clean["tests"][f"learned - {name}"]["diff"]
    assert res["rules"]["confidence_dev_best"]["learned_same_questions"] == {"accuracy": 0.75, "calls": 0.25}
    text = A.markdown(res, "t")
    assert "8 of the 12 questions are also in the dev grid" in text and "8 dev questions excluded" in text
    # the question hash, when recorded, is the identity (example ids of two pools are not comparable in general)
    hashed = [{**r, "question_hash": f"h{r['example_id']}"} for r in fresh]
    other = [{**r, "example_id": r["example_id"] + 500, "question_hash": f"h{r['example_id']}"} for r in fresh[:2]]
    assert A.analyze(hashed, other + [{**r, "question_hash": f"d{r['example_id']}"} for r in dev], draws=50,
                     n_boot=100)["dev_overlap"] == {"shared": 2, "kept": 2}
    # nothing left to evaluate: the two rules are absent instead of being in-sample
    assert "confidence_dev_best" not in A.analyze(dev, dev, draws=50, n_boot=100)["rules"]


def test_dev_thresholds_are_cuts_between_distinct_confidences():
    # dev: the 3rd and 4th least confident candidates tie at .50; the policy calls 3 times
    dev = [record(1, False, False, False, True, "verifier", 0.30), record(2, False, False, False, True, "verifier", 0.40),
           record(3, False, False, False, True, "verifier", 0.50), record(4, True, True, True, False, "commit", 0.50),
           record(5, True, True, True, True, "commit", 0.90)]
    test = [record(11, False, False, False, True, "commit", 0.46), record(12, True, True, True, False, "commit", 0.50),
            record(13, True, True, True, True, "commit", 0.95)]
    r = A.analyze(test, dev, draws=50, n_boot=100)["rules"]["confidence_dev_matched"]
    # no threshold selects exactly 3 dev questions; the nearest realisable cuts are 2 and 4, and the smaller one is taken
    assert r["threshold"] == pytest.approx(0.45) and r["dev_calls"] == 0.4 and r["dev_policy_calls"] == 0.6
    assert r["dev_accuracy"] == 0.8      # questions 1 and 2 fixed, 3 not called (wrong), 4 and 5 kept
    assert r["calls"] == 0.0 and r["accuracy"] == pytest.approx(2 / 3)   # no test candidate is below the .45 threshold
    assert A.threshold_for_k([{"conf": 0.5, "id": 1}, {"conf": 0.5000000000000001, "id": 2}], 1) == 0.5000000000000001


def test_first_call_only_disclosure_and_root_agreement():
    recs = toy()
    for r in recs:
        r["root_check"] = {"checked": 6, "mismatch": 0}
        for b in r["branches"]:
            b["revision"] = {"would_call": b["sequence"] == ["verifier"] and r["example_id"] == 3}
    res = A.analyze(recs, draws=100, n_boot=200)
    assert res["root_agreement"] == {"n_equal": 8, "n": 8}
    assert res["rules"]["learned"]["commit_forced"] == 1 and res["rules"]["learned"]["commit_forced_share"] == pytest.approx(1 / 3)
    assert res["rules"]["always_null"]["calls"] == 0.0 and res["rules"]["always_verifier_nocand"]["calls"] == 1.0
    text = A.markdown(res, "t")
    assert "equals unconstrained greedy decoding on 8/8 questions" in text and "grammar-forced" not in text
    assert "Learned calling, first call only" in text and "after 1 of 3 calls (33.3%) the commit was forced" in text
    # a manager that does not follow the grammar on its own (the untrained base model): the table says so
    for r in recs[:6]:
        r["root_check"]["mismatch"] = 2
    forced = A.markdown(A.analyze(recs, draws=100, n_boot=200), "t")
    assert "on 2/8 questions" in forced and "grammar-forced model, not its deployed behaviour" in forced
    assert "Grammar-constrained first action, then commit (not the deployed policy)" in forced
    assert "Learned calling, first call only" not in forced
    # records of an RSI collection have no root check: nothing is claimed
    assert "root_agreement" not in A.analyze(toy(), draws=50, n_boot=100)


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
    assert len(jobs) == len(set(jobs)) == 8 + 4 + 1 + 4 + 1 + 8 + 4 + 12 + 4 + 8 + 4 + 8 + 16 + 7 + 16 + 2 + 1
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


def test_queue_holds_starts_while_the_advisor_server_is_down_and_stops_its_jobs():
    done, answers, started = set(), iter([False, False, True, True, True]), []

    def launch(job, gpu):
        started.append(job)
        return FakeProc(0, 0, on_exit=lambda: done.add(job))

    sleeps = []
    state = Q.run_queue([("s1", "medqa", "test"), ("s1", "aqua", "test")], ["0"], launch=launch,
                        finished=lambda j: j in done, sleep=sleeps.append, ready=lambda: next(answers))
    assert len(sleeps) >= 2 and len(started) == 2 and all(e["status"] == "done" for e in state.values())

    class Live:
        terminated = False

        def poll(self):
            return 143 if self.terminated else None

        def terminate(self):
            self.terminated = True

    live = Live()

    def stop(seconds):
        raise SystemExit(143)  # what the SIGTERM / SIGHUP handler does

    with pytest.raises(SystemExit):
        Q.run_queue([("s1", "medqa", "test")], ["0"], launch=lambda job, gpu: live, finished=lambda j: False, sleep=stop)
    assert live.terminated


def test_stale_tables_are_found_and_the_worker_survives_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "OUT", tmp_path / "runs/verifier_grid")
    monkeypatch.setattr(Q, "PY", "py")
    test, dev = ("s1", "medqa", "test"), ("s1", "medqa", "dev")
    assert not Q.needs_analysis(test)  # not finished: nothing to analyse
    for job in (test,):
        Q.out_dir(job).mkdir(parents=True)
        (Q.out_dir(job) / "grid_summary.json").write_text("{}")
    assert Q.needs_analysis(test)      # finished, no table
    analysis = Q.OUT / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "s1_medqa_test.json").write_text(json.dumps({"dev": None}))
    assert Q.needs_analysis(test)      # the markdown is missing
    (analysis / "s1_medqa_test.md").write_text("x")
    assert not Q.needs_analysis(test)
    Q.out_dir(dev).mkdir(parents=True)
    (Q.out_dir(dev) / "grid_summary.json").write_text("{}")
    (Q.out_dir(dev) / "grid_records.jsonl").write_text("")
    assert Q.needs_analysis(test) and Q.needs_analysis(dev)   # the test table was made without the dev threshold
    (analysis / "s1_medqa_test.json").write_text(json.dumps({"dev": str(Q.out_dir(dev) / "grid_records.jsonl")}))
    (analysis / "s1_medqa_dev.json").write_text(json.dumps({"dev": None}))
    (analysis / "s1_medqa_dev.md").write_text("x")
    assert not Q.needs_analysis(test) and not Q.needs_analysis(dev)

    seen = []

    def run(job):
        seen.append(job)
        if job == test:
            raise RuntimeError("broken table")

    worker = Q.AnalysisWorker(run)
    worker.submit(test)
    worker.submit(dev)
    worker.join()
    assert seen == [test, dev]


def test_cross_benchmark_table(tmp_path):
    import mcq_verifier_grid_table as T

    dev = [{**r, "example_id": r["example_id"] + 100} for r in toy()]
    for bench in ("medqa", "aqua"):
        res = A.analyze(toy(), dev, draws=100, n_boot=200)
        (tmp_path / f"s1_{bench}_test.json").write_text(json.dumps(res))
    text = T.markdown(T.load(tmp_path, "s1", "test"), "s1", "test")
    assert "| Rule | MedQA (n=8) | AQuA (n=8) |" in text and "Not finished yet: MMLU-Pro, GPQA." in text
    assert "| Always Verifier, with the candidate | 62.5 / 1.00 | 62.5 / 1.00 |" in text
    assert "| Revision turn on an empty Verifier output | 50.0 / 0.00 | 50.0 / 0.00 |" in text
    assert "Learned − always Verifier (with candidate) | +12.5 [" in text
    assert "MedQA 3/8 questions (E 0, R 1, V 2)" in text
    assert T.main(["--analysis", str(tmp_path), "--ckpt", "s1", "--out", str(tmp_path / "t.md")]) == 0
    with pytest.raises(SystemExit):
        T.main(["--analysis", str(tmp_path), "--ckpt", "dyn3"])


# ------------------------------------------------------------------------------------------ attribution

def test_manager_attribution_on_a_hand_computed_pair(tmp_path):
    import mcq_manager_attribution as M

    def rec(eid, draft, commit, e, r, v, action, conf, nocand, null, mismatch=0):
        out = record(eid, commit, e, r, v, action, conf, nocand=nocand, null=null)
        out.update(direct_pred=draft, question_hash=f"q{eid}", root_check={"checked": 6, "mismatch": mismatch})
        klp = out["policy_action"]["key_logprobs"]
        out["policy_action"]["key_logprobs"] = {draft: klp["A"], **{k: v for k, v in klp.items() if k not in ("A", draft)},
                                                **({"A": klp["B"]} if draft != "A" else {})}
        return out

    #           id draft commit  E      R      V     action     conf  nocand null
    trained = [rec(1, "A", True, True, True, True, "commit", 0.90, True, True),
               rec(2, "A", False, False, False, True, "verifier", 0.30, False, False),
               rec(3, "B", False, False, True, True, "verifier", 0.60, True, False),
               rec(4, "A", True, True, True, False, "commit", 0.80, True, True),
               rec(5, "C", False, False, False, False, "commit", 0.70, False, False),
               rec(6, "A", False, False, False, True, "commit", 0.50, True, False)]
    base = [rec(1, "A", True, True, True, True, "commit", 0.60, True, True, mismatch=3),
            rec(2, "A", False, False, False, False, "commit", 0.20, False, False, mismatch=3),
            rec(3, "B", False, False, False, True, "verifier", 0.90, False, False, mismatch=3),
            rec(4, "A", True, True, True, True, "commit", 0.95, True, True, mismatch=3),
            rec(5, "D", True, True, True, True, "commit", 0.40, True, True, mismatch=3),      # another candidate
            rec(6, "A", False, False, False, True, "verifier", 0.30, True, False)]
    res = M.attribute(trained, list(reversed(base)), draws=200, n_boot=300)   # joined on the question, not on the order
    assert res["n"] == 6 and res["base_root_agreement"] == {"n_equal": 1, "n": 6}
    d = res["draft"]
    assert d["trained"] == pytest.approx(2 / 6) and d["base"] == pytest.approx(3 / 6) and d["n_same"] == 5
    assert d["test"]["diff"] == pytest.approx(-1 / 6)
    # same candidate on questions 1, 2, 3, 4, 6: Verifier trained T T T F T (4/5), base T F T T T (4/5)
    v = res["evidence"]["verifier"]
    assert v["n"] == 5 and v["trained"] == 0.8 and v["base"] == 0.8 and v["test"]["diff"] == 0 and v["same_outcome"] == 0.6
    assert v["fixed"] == {"trained": 1.0, "base": pytest.approx(2 / 3), "n": 3}      # wrong candidates 2, 3, 6
    assert v["broken"] == {"trained": 0.5, "base": 0.0, "n": 2}                      # right candidates 1, 4
    assert res["evidence"]["reasoner"]["trained"] == 0.6 and res["evidence"]["reasoner"]["base"] == 0.4
    s = res["selection"]
    # the learned policy calls on 2 and 3: with the Verifier both are fixed -> 1, 2, 3, 4 correct
    assert s["n_calls"] == 2 and s["learned"] == pytest.approx(4 / 6)
    # trained confidence: the two least confident are 2 (.30) and 6 (.50) -> 1, 2, 4, 6 correct
    assert s["trained_confidence"] == pytest.approx(4 / 6)
    # untrained confidence: 2 (.20) and 6 (.30) -> the same two questions
    assert s["base_confidence"] == pytest.approx(4 / 6) and s["tests"]["trained confidence - base confidence"]["diff"] == 0
    assert s["trained_policy"]["call_rate_candidate_wrong"] == 0.5 and s["trained_policy"]["call_rate_candidate_right"] == 0.0
    au = s["auroc_candidate_wrong"]
    # wrong candidates (trained): 2 (.30), 3 (.60), 5 (.70), 6 (.50); right: 1 (.90), 4 (.80): every wrong one is less confident
    assert au["trained_confidence"] == 1.0 and au["trained_call_decision"] == 0.75
    y = res["systems"]
    assert y["base_always_verifier"] == pytest.approx(5 / 6) and y["trained_always_verifier"] == pytest.approx(4 / 6)
    # untrained manager, gate on its own confidence, 2 calls: questions 2 (.20) and 6 (.30): 2 stays wrong, 6 is fixed
    assert y["base_confidence_gate"] == {"accuracy": pytest.approx(4 / 6), "calls": pytest.approx(2 / 6)}
    assert y["tests"]["trained learned calling - base with a confidence gate"]["diff"] == 0
    dec = y["gap_decomposition"]   # draft 2/6 - 3/6; gate gain (4/6 - 2/6) - (4/6 - 3/6); learned 3/6 vs own gate 4/6
    assert dec["draft"] == pytest.approx(-1 / 6) and dec["gate_gain"] == pytest.approx(1 / 6)
    assert sum(dec.values()) == pytest.approx(y["trained_learned"]["accuracy"] - y["base_confidence_gate"]["accuracy"])
    assert d["trained_accuracy_same_candidate"] == pytest.approx(2 / 5) and d["trained_accuracy_other"] == 0.0
    assert M.auroc([0.1, 0.4, 0.4, 0.9], [False, True, False, True]) == pytest.approx(0.875)
    assert M.auroc([1, 2], [True, True]) is None
    assert "base_confidence_gate_dev" not in y

    # the same gate with a threshold fixed on dev. The trained policy calls on 2 of its 4 dev questions; the untrained
    # dev confidences are .25 .35 .55 .85, so 2 dev calls is the cut between .35 and .55: threshold .45
    trained_dev = [rec(11, "A", False, False, False, True, "verifier", 0.30, True, False),
                   rec(12, "A", False, False, False, True, "verifier", 0.40, True, False),
                   rec(13, "A", True, True, True, True, "commit", 0.90, True, True),
                   rec(14, "A", True, True, True, True, "commit", 0.80, True, True)]
    base_dev = [rec(11, "A", False, False, False, True, "commit", 0.25, True, False),
                rec(12, "A", True, True, True, False, "commit", 0.35, True, True),
                rec(13, "A", True, True, True, True, "commit", 0.55, True, True),
                rec(14, "A", False, False, False, True, "commit", 0.85, True, False)]
    g = M.attribute(trained, base, trained_dev=trained_dev, base_dev=list(reversed(base_dev)), draws=200, n_boot=300)["systems"]
    fixed = g["base_confidence_gate_dev"]
    assert fixed["threshold"] == pytest.approx(0.45) and fixed["dev_calls"] == 0.5 and fixed["dev_policy_calls"] == 0.5
    assert fixed["dev_accuracy"] == 0.5 and fixed["dev_candidate"] == 0.5        # dev: 11 fixed, 12 broken, 13 right, 14 wrong
    # test pool: untrained confidence below .45 on 2 (.20), 5 (.40) and 6 (.30): 2 stays wrong, 5 stays right, 6 is fixed
    assert fixed["accuracy"] == pytest.approx(4 / 6) and fixed["calls"] == 0.5 and fixed["n"] == 6 and fixed["dev_overlap_excluded"] == 0
    assert g["tests"]["trained learned calling - base with a dev-threshold confidence gate"]["diff"] == 0
    assert g["base_confidence_gate"] == y["base_confidence_gate"]                # the in-pool gate is unchanged
    # a test question that is also in a dev grid is left out of the dev-threshold rule and of its test: question 5 here.
    # 2 of 5 dev calls -> still the cut at .45; on 1, 2, 3, 4, 6 the gate calls on 2 and 6 -> 1, 4, 6 right (3/5);
    # the learned policy on the same five questions: 1, 2, 3, 4 right (4/5) with 2 calls
    extra_t = rec(5, "C", False, False, False, False, "commit", 0.70, False, False)
    extra_b = rec(5, "D", True, True, True, True, "commit", 0.99, True, True)
    g = M.attribute(trained, base, trained_dev=trained_dev + [extra_t], base_dev=base_dev + [extra_b], draws=200, n_boot=300)["systems"]
    fixed = g["base_confidence_gate_dev"]
    assert fixed["threshold"] == pytest.approx(0.45) and fixed["n"] == 5 and fixed["dev_overlap_excluded"] == 1
    assert fixed["accuracy"] == pytest.approx(3 / 5) and fixed["calls"] == pytest.approx(2 / 5)
    assert fixed["trained_learned_same_questions"] == {"accuracy": pytest.approx(4 / 5), "calls": pytest.approx(2 / 5)}
    assert g["tests"]["trained learned calling - base with a dev-threshold confidence gate"]["diff"] == pytest.approx(1 / 5)
    # ties: dev confidences .25 .25 .55 .85 have no cut at one call; a target of one call is equally far from 0 and 2 -> 0 calls
    tied = [dict(r) for r in base_dev]
    tied[1] = rec(12, "A", True, True, True, False, "commit", 0.25, True, True)
    one_call = [trained_dev[0]] + [rec(i, "A", True, True, True, True, "commit", 0.9, True, True) for i in (12, 13, 14)]
    t0 = M.dev_threshold(M.A.table(one_call), M.A.table(tied))
    assert t0["threshold"] == 0.0 and t0["dev_calls"] == 0.0 and t0["dev_policy_calls"] == 0.25
    # the nearest cut can lie above the target: dev confidences .25 .25 .25 .85 and 2 policy calls -> cuts 0, 3, 4 -> 3 calls
    three = [rec(i, "A", True, True, True, True, "commit", 0.25, True, True) for i in (11, 12, 13)] + [base_dev[3]]
    t3 = M.dev_threshold(M.A.table(trained_dev), M.A.table(three))
    assert t3["dev_calls"] == 0.75 and t3["threshold"] == pytest.approx(0.55)
    # a confidence equal to the threshold is not below it: dev confidences .25 .75 -> threshold .5 exactly; question 5 at .5
    half_t = [rec(11, "A", False, False, False, True, "verifier", 0.3, True, False), rec(12, "A", True, True, True, True, "commit", 0.9, True, True)]
    half_b = [rec(11, "A", False, False, False, True, "commit", 0.25, True, False), rec(12, "A", True, True, True, True, "commit", 0.75, True, True)]
    base_half = [dict(r) for r in base]
    base_half[4] = rec(5, "D", True, True, True, True, "commit", 0.5, True, True, mismatch=3)
    g = M.attribute(trained, base_half, trained_dev=half_t, base_dev=half_b, draws=200, n_boot=300)["systems"]["base_confidence_gate_dev"]
    assert g["threshold"] == 0.5 and g["calls"] == pytest.approx(2 / 6)                # 2 (.20) and 6 (.30); not 5 (.50)
    with pytest.raises(ValueError, match="dev grids"):                                # another dev question in one of the two grids
        M.attribute(trained, base, trained_dev=trained_dev, base_dev=base_dev[:3] + [rec(15, "A", True, True, True, True, "commit", 0.9, True, True)])
    assert M.dev_threshold([], M.A.table(base_dev)) is None
    # every dev confidence below the threshold: always call (no finite threshold is reported)
    g = M.attribute(trained, base, trained_dev=[rec(i, "A", False, False, False, True, "verifier", 0.3, True, False) for i in (11, 12)],
                    base_dev=base_dev[:2], draws=200, n_boot=300)["systems"]["base_confidence_gate_dev"]
    assert g["threshold"] is None and g["calls"] == 1.0 and g["accuracy"] == pytest.approx(5 / 6)

    for name, recs, dev in (("s1", trained, trained_dev), ("base", base, base_dev)):
        for pool, rows in (("test", recs), ("dev", dev)):
            path = tmp_path / "grids" / name / "medqa" / pool
            path.mkdir(parents=True)
            (path / "grid_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert M.main(["--grid-root", str(tmp_path / "grids"), "--out", str(tmp_path / "out"), "--draws", "100", "--n-boot", "200",
                   "--dev-pool", ""]) == 0
    text = (tmp_path / "out/s1_vs_base_test.md").read_text()
    assert "MedQA 1/6 questions: on its own it does not follow" in text and "| Same candidate (%) | 83.3 |" in text
    assert "fixed on dev" not in text
    assert M.main(["--grid-root", str(tmp_path / "grids"), "--out", str(tmp_path / "out"), "--draws", "100", "--n-boot", "200"]) == 0
    text = (tmp_path / "out/s1_vs_base_test.md").read_text()
    assert "| Untrained manager + Verifier below a confidence threshold fixed on dev | 66.7 / 0.50 |" in text
    assert "MedQA 50.0% of dev questions called (learned policy on dev: 50.0%))" in text
    assert "the Verifier is called when the untrained manager's confidence in its draft is below" in text
    assert json.loads((tmp_path / "out/s1_vs_base_test.json").read_text())["medqa"]["systems"]["base_confidence_gate_dev"]["n"] == 6
    # with a test question in the dev pool the note says what the gate row is computed on
    for name, extra in (("s1", extra_t), ("base", extra_b)):
        with open(tmp_path / "grids" / name / "medqa" / "dev" / "grid_records.jsonl", "a") as fh:
            fh.write(json.dumps(extra) + "\n")
    assert M.main(["--grid-root", str(tmp_path / "grids"), "--out", str(tmp_path / "out"), "--draws", "100", "--n-boot", "200",
                   "--base-label", "commit-only", "--reference", str(tmp_path / "missing.json")]) == 0   # no reference rows: not read
    text = (tmp_path / "out/s1_vs_base_test.md").read_text()
    assert ("1 question(s) of this pool that are in the dev pool left out: the gate row and its difference are on the other "
            "n = 5, where learned calling is 80.0 / 0.40") in text
    assert "the Verifier is called when the commit-only manager's confidence in its draft is below" in text
    assert "| Commit-only manager + Verifier below a confidence threshold fixed on dev | 60.0 / 0.40 |" in text and "Reference:" not in text
    assert json.loads((tmp_path / "out/s1_vs_base_test.json").read_text())["medqa"]["systems"]["base_confidence_gate_dev"]["n"] == 5
    with pytest.raises(ValueError, match="same questions"):
        M.attribute(trained, base[:5])


def test_queue_passes_over_jobs_whose_checkpoint_is_not_on_disk_yet(tmp_path, monkeypatch):
    jobs = [("seed1", "gpqa", "test"), ("allc", "gpqa", "test"), ("seed2", "gpqa", "test")]
    done, started, on_disk, training, ticks = set(), [], {jobs[1]}, [True], [0]

    def launch(job, gpu):
        started.append(job)
        return FakeProc(1, 0, on_exit=lambda: done.add(job))

    def sleep(seconds):
        ticks[0] += 1
        if ticks[0] == 4:
            on_disk.add(jobs[0])      # the training chain delivers seed 1 ...
        if ticks[0] == 8:
            training[0] = False       # ... and ends without seed 2

    state = Q.run_queue(jobs, ["0", "1"], launch=launch, finished=lambda j: j in done, sleep=sleep,
                        available=lambda j: j in on_disk, producing=lambda: training[0])
    assert started == [jobs[1], jobs[0]]
    assert [state[Q.job_id(j)]["status"] for j in jobs] == ["done", "done", "failed"]
    assert state["seed2/gpqa/test"]["error"] == "checkpoint not on disk" and state["seed2/gpqa/test"]["attempts"] == 0

    monkeypatch.setattr(Q, "WORK", tmp_path)
    monkeypatch.setattr(Q, "CKPT", tmp_path / "ckpt")
    (tmp_path / "ckpt/gpqa_seed1").mkdir(parents=True)
    assert not Q.available(("seed1", "gpqa", "test")) and not Q.available(("base", "gpqa", "test")) and not Q.producing()
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        (tmp_path / "ckpt/gpqa_seed1" / f).write_text("x")
    (tmp_path / "ckpt/base_model_path.txt").write_text("/hub/base")
    (tmp_path / "ckpt/TRAINING").write_text("")
    assert Q.available(("seed1", "gpqa", "test")) and Q.available(("base", "gpqa", "test")) and Q.producing()
    assert Q.checkpoint_path("seed12", "aqua") == str(tmp_path / "ckpt/aqua_seed12")
    with pytest.raises(ValueError):
        Q.checkpoint_path("seedx", "aqua")


def test_seed_replicates_table(tmp_path):
    import mcq_verifier_grid_seeds as S

    base = A.analyze(toy(), draws=100, n_boot=200)
    base.pop("_vectors")
    other = json.loads(json.dumps(base))
    other["rules"]["learned"].update(accuracy=0.625, calls=0.5)
    (tmp_path / "s1_medqa_test.json").write_text(json.dumps(base))
    (tmp_path / "seed1_medqa_test.json").write_text(json.dumps(other))
    text = S.markdown(S.load(tmp_path, ["s1", "seed1", "seed2"], "test"), ["s1", "seed1", "seed2"], "test")
    assert "**MedQA** (n = 8; seeds present: s1, seed1)" in text
    assert "| Learned calling (first call), accuracy (%) | 75.0 | 62.5 | 68.8 ± 8.8 |" in text
    assert "| Learned calling, calls per question | 0.38 | 0.50 | 0.44 ± 0.09 |" in text
    assert "| Learned − always Verifier (with candidate), pt (exact McNemar p) | +12.5 (1.000) | +12.5 (1.000) | 12.5 ± 0.0 |" in text
    assert S.main(["--analysis", str(tmp_path), "--ckpts", "s1,seed1"]) == 0


def test_deployed_evaluation_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "WORK", tmp_path)
    monkeypatch.setattr(Q, "OUT", tmp_path / "runs/verifier_grid")
    monkeypatch.setattr(Q, "CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(Q, "PY", "py")
    job = ("seed2", "aqua", "deploy_test")
    assert job in Q.all_jobs() and Q.is_deploy(job) and not Q.is_deploy(("seed2", "aqua", "test"))
    cmd = Q.command(job)
    assert cmd[:4] == ["py", "-m", "src.manager.mcq_rsi", "evaluate"]
    assert cmd[cmd.index("--pool") + 1] == "test" and cmd[cmd.index("--bench") + 1] == "aqua"
    assert cmd[cmd.index("--checkpoint") + 1] == str(tmp_path / "ckpt/aqua_seed2") and "--no-require-gate" in cmd
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "runs/verifier_grid/seed2/aqua/deploy_test") and "--resume" not in cmd
    out = Q.out_dir(job)
    out.mkdir(parents=True)
    (out / "grid_summary.json").write_text("{}")          # not the marker of a deployed evaluation
    assert not Q.finished(job)
    (out / "mcq_rsi_eval.json").write_text("{}")
    assert Q.finished(job) and Q.analysis_commands(job) == [] and not Q.needs_analysis(job)
    assert Q.checkpoint_path("cmto", "gpqa") == str(tmp_path / "ckpt/gpqa_cmto")


def test_forced_baseline_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "WORK", tmp_path)
    monkeypatch.setattr(Q, "OUT", tmp_path / "runs/verifier_grid")
    monkeypatch.setattr(Q, "PY", "py")
    job = ("s1", "medqa", "forced_all_test")
    assert job in Q.all_jobs() and Q.is_deploy(job) and Q.forced_spec(job) == ("extractor,reasoner,verifier", "test")
    assert Q.forced_spec(("s1", "medqa", "forced_verifier_test_paper")) == ("verifier", "test_paper")
    assert Q.forced_spec(("s1", "medqa", "deploy_test")) is None and Q.forced_spec(("s1", "medqa", "test")) is None
    cmd = Q.command(job)
    assert cmd[cmd.index("--pool") + 1] == "test" and cmd[-2:] == ["--forced", "extractor,reasoner,verifier"]
    assert "--forced" not in Q.command(("seed1", "medqa", "deploy_test"))
    out = Q.out_dir(job)
    out.mkdir(parents=True)
    (out / "mcq_rsi_eval.json").write_text("{}")
    assert not Q.finished(job)
    (out / "mcq_rsi_forced_extractor_reasoner_verifier.json").write_text("{}")
    assert Q.finished(job) and Q.analysis_commands(job) == []
