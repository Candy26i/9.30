"""MCQ RSI controller (design §3.2, §3.6, §7.1 items 8 and 10): plan, signature, deadline, gates, final test, e2e."""
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mcq_rsi_helpers import FakeManager, advisor_server, make_rows, tiny_model, tiny_tokenizer
from src.manager.marginal_value import _make_sft_rows
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import collect as C
from src.manager.mcq_rsi import controller as CT
from src.manager.mcq_rsi.advisors import CachedAdvisorPool

ROOT = Path(__file__).resolve().parents[1]
MEDQA = registry.get("medqa")


def cfg_for(tmp_path, **over):
    raw = {"bench": "medqa", "import_dir": str(tmp_path / "import"), "advisor_cache": str(tmp_path / "cache"),
           "advisor_url": None, "preflight": {"required": False}}
    raw.update(over)
    return CT.load_config(raw)


# ------------------------------------------------------------------------------- planning

def test_plan_shape_dedup_static_reuse_and_sft_init_chain(tmp_path):
    cfg = cfg_for(tmp_path)
    plan = CT.build_plan(cfg, "main", ["dynamic", "static", "success", "dynamic_sft"], 3)
    names = [s["name"] for s in plan["stages"]]
    assert len(names) == len(set(names))
    by = {s["name"]: s for s in plan["stages"]}
    # Round 1 is one shared set of stages, before any round-2 stage.
    assert names[:4] == ["prefetch/advisors", "r1/S1_dev", "r1/grpo", "r1/grpo_dev"]
    assert sum(n.startswith("r1/") for n in names) == 3
    arms = plan["by_arm"]
    # Round-2 collection from the shared G_1 is one stage for dynamic and success; dynamic_sft (no GRPO) starts from S_1.
    assert arms["dynamic"]["2"]["collect"] == arms["success"]["2"]["collect"] == "r2/collect"
    assert by["r2/collect"]["params"]["checkpoint"] == "decision:r1/grpo_dev"
    # A rejected G_1 falls back to the imported S_1 (round 1 has no round SFT).
    assert {k: by["r1/grpo_dev"]["params"][k] for k in ("fallback", "baseline", "grpo_stage", "role")} == {
        "fallback": "import:S_1", "baseline": "r1/S1_dev", "grpo_stage": "r1/grpo", "role": "grpo"}
    assert by["r1/grpo"]["params"]["anchor"] == "import:labels"
    assert arms["dynamic_sft"]["2"]["collect"] == "r2/dynamic_sft/collect"
    assert by["r2/dynamic_sft/collect"]["params"]["checkpoint"] == "decision:r1/S1_dev"
    assert arms["dynamic"]["3"]["collect"] != arms["success"]["3"]["collect"]
    # Static reuses the round-1 label file in every round, through one shared select stage, and never collects.
    assert arms["static"]["2"]["select"] == arms["static"]["3"]["select"] == "static/select"
    assert by["static/select"]["params"] == {"selection": "static", "records": None}
    assert "collect" not in arms["static"]["2"] and "collect" not in arms["static"]["3"]
    assert by["r2/static/sft"]["params"]["labels"] == by["r3/static/sft"]["params"]["labels"] == "out:static/select::labels.jsonl"
    # SFT_k is initialised from G_(k-1); GRPO_k starts from S_k and anchors on the round's labels.
    for arm in ("dynamic", "static", "success"):
        assert by[f"r2/{arm}/sft"]["params"]["init"] == "decision:r1/grpo_dev"
        assert by[f"r3/{arm}/sft"]["params"]["init"] == f"decision:r2/{arm}/grpo_dev"
        assert by[f"r3/{arm}/grpo"]["params"]["checkpoint"] == f"out:r3/{arm}/sft::model"
        assert by[f"r3/{arm}/grpo"]["params"]["anchor"] == by[f"r3/{arm}/sft"]["params"]["labels"]
        assert by[f"r3/{arm}/grpo_dev"]["params"]["fallback"] == f"out:r3/{arm}/sft::model"
        assert plan["finals"][arm] == f"decision:r3/{arm}/grpo_dev"
    assert by["r3/dynamic_sft/sft"]["params"]["init"] == "decision:r2/dynamic_sft/sft_dev"
    assert "grpo" not in arms["dynamic_sft"]["3"] and plan["finals"]["dynamic_sft"] == "decision:r3/dynamic_sft/sft_dev"
    assert by["r1/grpo"]["lane"] == "train" and by["r2/collect"]["lane"] == "inference" and by["static/select"]["lane"] == "cpu"
    # Every reference names a stage that comes earlier in the plan.
    order = {n: i for i, n in enumerate(names)}
    for i, s in enumerate(plan["stages"]):
        for v in s["params"].values():
            if isinstance(v, str) and v.split(":", 1)[0] in ("out", "decision"):
                ref = v.split(":", 1)[1].split("::")[0].split("#")[0]
                assert order[ref] < i, (s["name"], v)
    # Arms subset and rounds.
    small = CT.build_plan(cfg, "main", ["static"], 1)
    assert [s["name"] for s in small["stages"]] == ["prefetch/advisors", "r1/S1_dev", "r1/grpo", "r1/grpo_dev"]
    with pytest.raises(ValueError):
        CT.build_plan(cfg, "main", ["dynamic", "dynamic"], 2)
    with pytest.raises(ValueError):
        CT.build_plan(cfg, "main", ["dynamic"], 4)


def test_planner_dedups_identical_content_and_refuses_name_reuse():
    P = CT.Planner()
    a = P.add("x/a", "select", "cpu", selection="static", records=None)
    assert P.add("y/b", "select", "cpu", selection="static", records=None) == a and len(P.stages) == 1
    with pytest.raises(ValueError, match="reused"):
        P.add("x/a", "select", "cpu", selection="dynamic", records="out:z::r")


def test_pilot_plan_sweeps_round1_lr_then_dynamic_r2(tmp_path):
    plan = CT.build_plan(cfg_for(tmp_path), "pilot")
    assert plan["arms"] == ["dynamic"] and plan["rounds"] == 2
    by = {s["name"]: s for s in plan["stages"]}
    assert [by[f"r1/grpo_lr{t}"]["params"]["learning_rate"] for t in ("2e-06", "5e-06", "1e-05")] == [2e-6, 5e-6, 1e-5]
    assert by["r1/grpo_select"]["kind"] == "grpo_select" and len(by["r1/grpo_select"]["params"]["candidates"]) == 3
    assert by["r2/collect"]["params"]["checkpoint"] == "decision:r1/grpo_select"
    assert by["r2/dynamic/grpo"]["params"]["learning_rate_ref"] == "decision:r1/grpo_select#learning_rate"
    assert "r1/grpo" not in by


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="unknown config key"):
        cfg_for(tmp_path, typo=1)
    with pytest.raises(ValueError, match="unknown config key"):
        cfg_for(tmp_path, gates={"acc_tol": 0.1})
    with pytest.raises(ValueError, match="FA-GRPO"):
        cfg_for(tmp_path, grpo={"stepz": 3})
    with pytest.raises(KeyError):
        CT.load_config({"bench": "nope"})
    # The advisor server is configured by scripts/start_mcq_advisors.sh (one server for every benchmark), not by
    # a config section: every config's advisor_url must point at the scripts' default port.
    with pytest.raises(ValueError, match="unknown config key"):
        cfg_for(tmp_path, server={"port": 1})
    port = re.search(r'MCQ_ADVISOR_PORT:-(\d+)', (ROOT / "scripts" / "start_mcq_advisors.sh").read_text()).group(1)
    assert port == re.search(r'MCQ_ADVISOR_PORT:-(\d+)', (ROOT / "scripts" / "runpod_mcq_rsi.sh").read_text()).group(1)
    for name in ("medqa", "mmlu_pro", "gpqa", "aqua", "smoke"):
        cfg = CT.load_config(ROOT / "configs" / f"mcq_rsi_{name}.json")
        assert cfg["advisor_url"] == f"http://127.0.0.1:{port}" and cfg["gpus"] == {"inference": "0", "train": "1"}
        assert cfg["grpo"]["learning_rate"] == 5e-6 and cfg["sft"]["num_train_epochs"] == 3
        assert cfg["pilot"]["rounds"] == 2 and cfg["select"]["rho"] is None
        CT.build_plan(cfg, "main")
        CT.build_plan(cfg, "pilot")
    smoke = CT.load_config(ROOT / "configs" / "mcq_rsi_smoke.json")
    assert smoke["final"]["test_pools"] == [] and smoke["limits"]["dev"] == 50 and smoke["grpo"]["steps"] == 4
    assert CT.load_config(ROOT / "configs" / "mcq_rsi_mmlu_pro.json")["final"]["test_pools"] == ["test", "test_paper"]
    for name in ("medqa", "mmlu_pro", "gpqa", "aqua"):  # every first role has a forced eval for the replay
        forced = CT.load_config(ROOT / "configs" / f"mcq_rsi_{name}.json")["final"]["forced"]
        assert {"extractor", "reasoner", "verifier"} <= set(forced) and "extractor,reasoner,verifier" in forced


# ---------------------------------------------------------------------- signature / deadline

def test_signature_refuses_changed_settings_and_code(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    run, root = CT.prepare_run(cfg, tmp_path / "run")
    CT.start(run, root)
    sig = json.loads((root / CT.RUN_FILE).read_text())["signature"]
    assert sig["code"]["git_head"] == CT.git_head()
    files = sig["code"]["files"]
    for rel in ("manager/mcq_rsi/controller.py", "manager/mcq_rsi/grpo.py", "manager/mcq_rsi/prompts/medqa/reasoner.txt",
                "pipeline/stages.py", "manager/evolve.py", "manager/routing_anchor.py", "subagents/runtime.py",
                "manager/prompt.py", "subagents/train.py", "subagents/prompts/runtime_prompts.py", "benchmarks/base.py",
                "benchmarks/aqua_rat.py", "benchmarks/medqa.py", "utils/seed.py", "utils/cache.py", "utils/leakage.py",
                "subagents/prompts/verifier.py", "teachers/base.py", "verifiable/actions.py"):
        assert rel in files, rel
    assert sig["manifests"]["split_manifest_sha256"] and "harness" in sig["code"]
    # Same settings resume; operational settings may change; anything else is refused.
    CT.start(CT.prepare_run({**cfg, "advisor_url": "http://127.0.0.1:18002", "advisor_workers": 8}, tmp_path / "run")[0], root)
    with pytest.raises(ValueError, match="run settings changed"):
        CT.start(CT.prepare_run({**cfg, "grpo": {"learning_rate": 1e-5}}, tmp_path / "run")[0], root)
    with pytest.raises(ValueError, match="run settings changed"):
        CT.start(CT.prepare_run(cfg, tmp_path / "run", arms=["dynamic"])[0], root)
    real = CT.code_identity

    def changed():
        ident = real()
        ident["files"]["manager/mcq_rsi/select.py"] = "0" * 64
        return ident

    monkeypatch.setattr(CT, "code_identity", changed)
    with pytest.raises(ValueError, match="code"):
        CT.start(CT.prepare_run(cfg, tmp_path / "run")[0], root)
    with pytest.raises(RuntimeError, match="code/manifests changed"):
        CT.check_code_unchanged(json.loads((root / CT.RUN_FILE).read_text()))
    from src.manager.mcq_rsi import __main__ as cli
    with pytest.raises(RuntimeError, match="code/manifests changed"):  # stage subprocesses refuse too
        cli.main(["stage", "--run-dir", str(root), "--name", "r1/S1_dev"])
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "x").write_text("x")
    monkeypatch.setattr(CT, "code_identity", real)
    with pytest.raises(ValueError, match="non-empty"):
        CT.start(CT.prepare_run(cfg, tmp_path / "other")[0], tmp_path / "other")


def test_deadline_persists_across_restarts(tmp_path, capsys):
    b = CT.persistent_deadline(tmp_path, 2, now=1000.0)
    assert b["deadline_unix"] == 1000.0 + 7200
    again = CT.persistent_deadline(tmp_path, 72, now=999999.0)
    assert again == b and "persisted deadline kept" in capsys.readouterr().out
    for bad in (0, -1, 72.5, 100):
        with pytest.raises(ValueError):
            CT.persistent_deadline(tmp_path / "x", bad)
    assert CT.persistent_deadline(tmp_path / "y", 72, now=0.0)["deadline_unix"] == 72 * 3600


def test_deadline_stops_the_run_and_keeps_completed_stages(tmp_path):
    cfg = cfg_for(tmp_path)
    root = tmp_path / "run"
    run, root = CT.prepare_run(cfg, root)
    CT.start(run, root)
    _write = CT._write_json
    _write(root / "budget.json", {"hours": 1.0, "started_unix": 0.0, "deadline_unix": time.time() - 1})
    with pytest.raises(TimeoutError):
        CT.run(cfg, root, hours=1, executor="inprocess", rt=None)
    st = json.loads((root / "status.json").read_text())
    assert st["controller"] == "deadline" and st["current_stage"] == "prefetch/advisors"


def test_run_lock_is_exclusive(tmp_path):
    with CT.run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="in use"):
            with CT.run_lock(tmp_path):
                pass
    with CT.run_lock(tmp_path):  # released
        pass


def test_subprocess_executor_kills_at_deadline_and_reports_failures(tmp_path):
    log = tmp_path / "log.txt"
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        CT.run_subprocess([sys.executable, "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(60)"],
                          log, dict(os.environ), time.time() + 1.5, poll=0.2)
    assert time.monotonic() - start < 30
    pid = int(re.findall(r"^(\d+)$", log.read_text(), re.M)[-1])
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the stage's process group was killed
    with pytest.raises(RuntimeError, match="exit 3"):
        CT.run_subprocess([sys.executable, "-c", "raise SystemExit(3)"], log, dict(os.environ), time.time() + 60, poll=0.1)
    assert CT.run_subprocess([sys.executable, "-c", "print('ok')"], log, dict(os.environ), time.time() + 60, poll=0.1) >= 0
    assert "$ " in log.read_text()
    env = CT.stage_env({"gpus": {"inference": "0", "train": "1"}}, "train")
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["PYTHONUNBUFFERED"] == "1"
    cmd = CT.stage_command(tmp_path, "r2/dynamic/sft")
    assert cmd[1:4] == ["-m", "src.manager.mcq_rsi", "stage"] and cmd[-2:] == ["--name", "r2/dynamic/sft"]


# ------------------------------------------------------------------------------------ gates

def _metrics(acc, calls, gap, rate=None):
    return {"accuracy": acc, "calls_per_example": calls, "call_rate": calls if rate is None else rate, "call_gap": gap}


def test_parity_gate_targets_and_aqua_call_ambiguity(tmp_path):
    cfg = cfg_for(tmp_path)
    ok = CT.parity_check(MEDQA, _metrics(0.835, 0.43, 0.30), cfg)
    assert ok["status"] == "pass" and {c["target"] for c in ok["checks"]} == {"dev_accuracy", "dev_avg_tool_calls", "dev_call_gap"}
    assert not next(c for c in ok["checks"] if c["target"] == "dev_call_gap")["gated"]  # reported, not gated
    assert CT.parity_check(MEDQA, _metrics(0.79, 0.405, 0.47), cfg)["status"] == "fail"  # -3 pt
    assert CT.parity_check(MEDQA, _metrics(0.82, 0.48, 0.47), cfg)["status"] == "fail"  # +0.075 calls
    assert CT.parity_check(MEDQA, _metrics(0.79, 0.405, 0.47), cfg, subset=True)["status"] == "subset"
    aqua = registry.get("aqua")
    # Paper "Calls" 0.894 may be call rate or calls/example: either within 0.05 passes.
    assert CT.parity_check(aqua, _metrics(0.78, 1.10, 0.1, rate=0.90), cfg)["status"] == "pass"
    assert CT.parity_check(aqua, _metrics(0.78, 0.90, 0.1, rate=0.60), cfg)["status"] == "pass"
    assert CT.parity_check(aqua, _metrics(0.78, 1.10, 0.1, rate=0.60), cfg)["status"] == "fail"
    assert CT.parity_check(registry.get("gpqa"), _metrics(0.5, 0.5, 0.0), cfg)["status"] == "not_applicable"
    assert CT.parity_check(registry.get("gpqa"), _metrics(0.55, 0.52, 0.0), cfg, "test")["status"] == "pass"
    mmlu = CT.parity_check(registry.get("mmlu_pro"), _metrics(0.645, 0.36, 0.22, rate=0.34), cfg)
    assert mmlu["status"] == "pass" and len([c for c in mmlu["checks"] if c["gated"]]) == 3


class StubRuntime(CT.Runtime):
    def import_path(self, what):
        return Path(self.cfg["import_dir"]) / ("S_1" if what == "S_1" else "labels.jsonl")


def _eval_dir(root, name, metrics, decision=None):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "mcq_rsi_eval.json").write_text(json.dumps({"metrics": metrics, "gate": [], "passed": True}))
    if decision is not None:
        (d / "decision.json").write_text(json.dumps({**decision, "dev_result": str(d / "mcq_rsi_eval.json"),
                                                     "metrics": metrics}))
    return d


def test_rejected_grpo_propagates_sk_and_accepted_grpo_advances(tmp_path, monkeypatch):
    from src.manager.mcq_rsi import evaluate as E
    cfg = cfg_for(tmp_path)
    rt = StubRuntime(cfg)
    root = tmp_path / "run"
    s_k = root / "r2/dynamic/sft/model"
    _eval_dir(root, "r2/dynamic/sft_dev", _metrics(0.80, 0.40, 0.40), {"checkpoint": str(s_k), "role": "sft"})
    for name, info in (("r2/dynamic/grpo", True), ("r2/dynamic/grpo_b", False)):
        (root / name).mkdir(parents=True)
        (root / name / "summary.json").write_text(json.dumps({"informativeness": {"passed": info}, "steps": 64,
                                                              "selected_step": 64, "accepted_by_guard": True}))
    outcomes = {}

    def fake_eval(checkpoint, rows, pool, out, **kw):
        m = outcomes[str(out)]
        Path(out, "mcq_rsi_eval.json").write_text(json.dumps({"metrics": m, "gate": [], "passed": True}))
        return {"metrics": m, "gate": [], "passed": True}

    monkeypatch.setattr(E, "evaluate", fake_eval)
    monkeypatch.setattr(StubRuntime, "rows", lambda self, pool: [])
    monkeypatch.setattr(StubRuntime, "pool", lambda self: None)

    def grpo_dev(name, grpo_stage, metrics):
        outcomes[str(root / name)] = metrics
        spec = {"name": name, "kind": "eval", "params": {"checkpoint": f"out:{grpo_stage}::final", "pool": "dev",
                                                          "role": "grpo", "baseline": "r2/dynamic/sft_dev",
                                                          "grpo_stage": grpo_stage, "fallback": "out:r2/dynamic/sft::model"}}
        (root / name).mkdir(parents=True, exist_ok=True)
        return CT.stage_eval(root, rt, spec, root / name)

    # Accuracy -2 pt: rejected -> G_2 := S_2 and its dev result is S_2's.
    rej = grpo_dev("r2/dynamic/grpo_dev", "r2/dynamic/grpo", _metrics(0.78, 0.40, 0.40))
    assert not rej["accept"]["accepted"] and rej["accept"]["decision"] == "grpo_rejected"
    assert rej["checkpoint"] == str(s_k) and rej["rejected_checkpoint"].endswith("r2/dynamic/grpo/final")
    assert rej["dev_result"].endswith("r2/dynamic/sft_dev/mcq_rsi_eval.json")
    # ... which is what round 3 resolves for its collection and SFT init.
    assert CT.resolve(root, rt, "decision:r2/dynamic/grpo_dev") == str(s_k)
    # Within tolerance and informative: accepted -> G_2 is the GRPO adapter.
    acc = grpo_dev("r2/dynamic/grpo_dev2", "r2/dynamic/grpo", _metrics(0.795, 0.50, 0.25))
    assert acc["accept"]["accepted"] and acc["checkpoint"].endswith("r2/dynamic/grpo/final")
    # Uninformative GRPO is rejected even with better dev numbers.
    bad = grpo_dev("r2/dynamic/grpo_dev3", "r2/dynamic/grpo_b", _metrics(0.9, 0.40, 0.40))
    assert not bad["accept"]["accepted"] and "uninformative" in bad["accept"]["reasons"][0]
    # SFT flags against the previous round's resolved G (never a selection).
    outcomes[str(root / "r3/dynamic/sft_dev")] = _metrics(0.70, 1.6, 0.1)
    spec = {"name": "r3/dynamic/sft_dev", "kind": "eval", "params": {"checkpoint": "out:r3/dynamic/sft::model",
                                                                      "pool": "dev", "role": "sft",
                                                                      "previous": "decision:r2/dynamic/grpo_dev"}}
    (root / spec["name"]).mkdir(parents=True)
    flags = CT.stage_eval(root, rt, spec, root / spec["name"])
    assert len(flags["flags"]) == 2 and flags["previous_checkpoint"] == str(s_k)
    assert flags["checkpoint"].endswith("r3/dynamic/sft/model")


def test_pilot_lr_selection(tmp_path):
    cfg = cfg_for(tmp_path)
    rt = StubRuntime(cfg)
    root = tmp_path / "run"
    s1 = _eval_dir(root, "r1/S1_dev", _metrics(0.80, 0.4, 0.4), {"checkpoint": "S1"})

    def cand(lr, accepted, acc, calls):
        d = _eval_dir(root, f"r1/g{lr}_dev", _metrics(acc, calls, 0.4))
        (d / "decision.json").write_text(json.dumps({
            "checkpoint": f"G@{lr}" if accepted else "S1", "dev_result": str(d / "mcq_rsi_eval.json") if accepted
            else str(s1 / "mcq_rsi_eval.json"), "accept": {"accepted": accepted, "reasons": [] if accepted else ["x"]}}))
        return {"learning_rate": lr, "grpo": f"r1/g{lr}", "dev": f"r1/g{lr}_dev"}

    def select(cands, name):
        spec = {"name": name, "kind": "grpo_select",
                "params": {"candidates": cands, "baseline": "r1/S1_dev", "default_learning_rate": None}}
        (root / name).mkdir(parents=True)
        return CT.stage_grpo_select(root, rt, spec, root / name)

    # 1e-5 has the best dev accuracy but was rejected by the §3.6 rule; among accepted, best accuracy wins.
    d = select([cand(2e-6, True, 0.80, 0.40), cand(5e-6, True, 0.82, 0.45), cand(1e-5, False, 0.90, 0.9)], "sel1")
    assert d["learning_rate"] == 5e-6 and d["checkpoint"] == "G@5e-06" and d["selected"] == "r1/g5e-06"
    # Accuracy tie: fewer calls, then the smaller lr.
    d = select([cand(3e-6, True, 0.81, 0.50), cand(4e-6, True, 0.81, 0.42), cand(6e-6, True, 0.81, 0.42)], "sel2")
    assert d["learning_rate"] == 4e-6
    # Nothing accepted: G_1 := S_1 and later rounds use the config lr.
    d = select([cand(7e-6, False, 0.9, 0.4)], "sel3")
    assert d["checkpoint"] == "S1" and d["selected"] is None and d["learning_rate"] == cfg["grpo"].get("learning_rate", 5e-6)
    assert CT.resolve(root, rt, "decision:sel1#learning_rate") == 5e-6


def test_ack_and_retry(tmp_path):
    cfg = cfg_for(tmp_path)
    run, root = CT.prepare_run(cfg, tmp_path / "run")
    CT.start(run, root)
    with pytest.raises(ValueError):
        CT.ack_gate(root, "r1/S1_dev", " ")
    assert CT.ack_gate(root, "r1/S1_dev", "known 2.5 pt shift")["r1/S1_dev"]["reason"] == "known 2.5 pt shift"
    (root / "r1" / "S1_dev").mkdir(parents=True)
    (root / "r1" / "S1_dev" / "partial").write_text("x")
    moved = CT.retry_stage(root, "r1/S1_dev")
    assert moved.name.startswith("S1_dev.failed-") and (moved / "partial").exists() and not (root / "r1/S1_dev").exists()
    # A completed stage is never moved aside.
    spec = next(s for s in run["plan"]["stages"] if s["name"] == "r1/S1_dev")
    _eval_dir(root, "r1/S1_dev", _metrics(0.80, 0.4, 0.4), {"checkpoint": "S1", "role": "s1"})
    (root / "r1/S1_dev" / CT.MARKER).write_text(json.dumps({"spec_sha256": "x"}))
    with pytest.raises(RuntimeError, match="completed stages are never redone"):
        CT.retry_stage(root, "r1/S1_dev")
    assert (root / "r1/S1_dev" / CT.MARKER).exists()


def test_final_test_refuses_before_completion_and_registers_truncated_finals(tmp_path):
    cfg = cfg_for(tmp_path)
    run, root = CT.prepare_run(cfg, tmp_path / "run", arms=["dynamic"], rounds=2)
    CT.start(run, root)
    rt = StubRuntime(cfg)
    with pytest.raises(RuntimeError, match="not complete"):
        CT.final_test(root, rt=rt)
    # Round 1 finished (G_1 accepted); round 2 did not.
    d = root / "r1/grpo_dev"
    d.mkdir(parents=True)
    (d / "decision.json").write_text(json.dumps({"checkpoint": str(root / "r1/grpo/final")}))
    (d / CT.MARKER).write_text("{}")
    reg = CT.final_test(root, rt=rt, accept_incomplete="pod lost", dry_run=True)
    assert reg["finals"]["dynamic"] == {"ref": "decision:r1/grpo_dev", "checkpoint": str(root / "r1/grpo/final"),
                                        "round": 1, "truncated_at_round": 1}
    assert reg["finals"]["S_1"]["checkpoint"] == str(Path(cfg["import_dir"]) / "S_1")
    names = [s["name"] for s in reg["stages"]]
    forced = ["extractor", "reasoner", "verifier", "extractor+reasoner+verifier"]
    assert names == ["final/S_1/test", *[f"final/S_1/dev_forced_{t}" for t in forced],
                     "final/dynamic/test", *[f"final/dynamic/dev_forced_{t}" for t in forced]]
    assert not (root / "final" / "finals.json").exists()  # a dry run registers nothing


# ------------------------------------------------------------------------- end to end (CPU)

COMMIT = "DRAFT_ANSWER_{0}\nANSWER_{0}".format
CALL_V = "DRAFT_ANSWER_{0}\n\n<tool_call>\n<function=verifier_tool>\n<parameter=current_draft>\n{0}\n</parameter>\n</function>\n</tool_call>".format
OFFSET = 10_000_000


class ScriptTok:
    def __init__(self, tok):
        self.tok, self.texts = tok, []
        self.pad_token_id, self.eos_token_id = tok.pad_token_id, tok.eos_token_id

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
    """Scripted eval manager: every gold is answered after one Verifier call on wrong drafts; GRPO adapters
    of round 2 commit their (wrong) draft everywhere, so their dev accuracy drops and the gate rejects them."""

    def __init__(self, tok, manager_dir):
        self.tok, self.bad = tok, "r2/dynamic/grpo" in str(manager_dir)

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, **kw):
        import torch
        prompt = self.tok.tok.decode(input_ids[0].tolist())
        eid = int(re.search(r"Example ID: (\d+)", prompt).group(1))
        keys = "ABCD"
        gold = keys[eid % 4]  # make_rows: ground truth keys[i % 4] with ids start + i, start % 4 == 0
        draft = gold if eid % 2 == 0 else keys[(eid + 1) % 4]
        if self.bad:
            text = COMMIT(draft)
        elif "<tool_response>" in prompt:
            text = COMMIT(gold)
        else:
            text = COMMIT(draft) if draft == gold else CALL_V(draft)
        self.tok.texts.append(text)
        return torch.cat([input_ids, torch.tensor([[OFFSET + len(self.tok.texts) - 1]])], 1)


@pytest.fixture
def tiny_world(tmp_path, monkeypatch):
    """Tiny Qwen3.5 base, S_1 LoRA + tokenizer + round-1 labels in an import dir, fake advisors, scripted eval."""
    import torch
    from peft import LoraConfig, get_peft_model
    from src.pipeline import stages
    torch.set_num_threads(1)
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    model.save_pretrained(tmp_path / "base")
    s1 = tmp_path / "import" / "medqa" / "round1" / "sft"
    get_peft_model(model, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, init_lora_weights=False,
                                     target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"])).save_pretrained(s1)
    tok.save_pretrained(s1)
    http, adapters = advisor_server(tmp_path)
    pool = CachedAdvisorPool("medqa", tmp_path / "cache", "http://x", adapters=adapters, http=http, sleep=lambda s: None)
    rows = make_rows(4)
    e = [r["example_id"] for r in rows]
    script = {e[0]: {"root": ("A", "commit")}, e[1]: {"root": ("A", "verifier"), "rev": {("extractor",): "B"}},
              e[2]: {"root": ("C", "commit")}, e[3]: {"root": ("A", "commit"), "rev": {("verifier",): "D"}}}
    labels = []
    for r in rows:
        labels += [{**x, "split": "train"} for x in _make_sft_rows(C.collect_question(r, MEDQA, FakeManager(script), pool,
                                                                                         max_depth=1))]
    (s1.parent / "labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in labels))
    script_tok = ScriptTok(tok)
    monkeypatch.setattr(stages, "_load_manager_for_eval", lambda ctx, manager_dir, device, dtype:
                        (script_tok, ScriptModel(script_tok, manager_dir)))
    pools = {"dev": make_rows(4, start=100), "test": make_rows(4, start=500), "collect_r2": make_rows(8, start=200),
             "grpo_r1": make_rows(4, start=300), "grpo_r2": make_rows(4, start=400)}

    keys = "ABCD"
    collect_script = {200 + i: ({"root": (keys[i % 4], "commit")} if i % 2 == 0 else
                                {"root": (keys[(i + 1) % 4], "commit"), "rev": {("verifier",): keys[i % 4]}})
                      for i in range(8)}  # even roots correct, odd roots rescued by the Verifier only

    class World(CT.Runtime):
        def pool(self):
            return pool

        def manager(self, checkpoint):  # scripted decisions so the round-2 labels hold commits and rescues
            return FakeManager(collect_script, name=str(checkpoint))

        def rows(self, name):
            return [dict(r, split="train" if name.startswith(("collect", "grpo")) else name) for r in pools[name]]

        def base(self):
            return str(tmp_path / "base")

    raw = {"bench": "medqa", "base_model": str(tmp_path / "base"), "base_revision": None,
           "import_dir": str(tmp_path / "import"), "advisor_cache": str(tmp_path / "cache"), "rounds": 2,
           "arms": ["dynamic"], "limits": {"dev": 4, "test": 4, "collect": 8, "grpo": 4},
           "prefetch": {"enabled": True, "verifier_pools": ["dev"]},
           "sft": {"max_steps": 2, "gradient_accumulation_steps": 1, "bf16": False},
           "grpo": {"steps": 2, "questions_per_step": 2, "guard_every": 1, "guard_probe_size": 2, "device": "cpu",
                    "learning_rate": 5e-3},
           "final": {"test_pools": ["test"], "forced": ["verifier"]}, "preflight": {"required": False}}
    cfg = CT.load_config(raw)
    return World(cfg), cfg, pool


def test_end_to_end_two_rounds_one_arm_four_questions(tmp_path, tiny_world, monkeypatch):
    rt, cfg, pool = tiny_world
    started = time.monotonic()
    root = tmp_path / "run"
    result = CT.run(cfg, root, hours=1, executor="inprocess", rt=rt)
    assert result["complete"] and result["completed_stages"] == result["planned_stages"] == 10
    names = [s["name"] for s in json.loads((root / CT.RUN_FILE).read_text())["plan"]["stages"]]
    assert names == ["prefetch/advisors", "r1/S1_dev", "r1/grpo", "r1/grpo_dev", "r2/collect", "r2/dynamic/select",
                     "r2/dynamic/sft", "r2/dynamic/sft_dev", "r2/dynamic/grpo", "r2/dynamic/grpo_dev"]
    assert all((root / n / CT.MARKER).is_file() for n in names)
    # Round 1: GRPO accepted (dev unchanged, informative) -> G_1 = the GRPO adapter, used by collection and SFT_2.
    g1 = json.loads((root / "r1/grpo_dev/decision.json").read_text())
    assert g1["accept"]["accepted"] and g1["checkpoint"] == str(root / "r1/grpo/final")
    col = json.loads((root / "r2/collect/collect_manifest.json").read_text())
    assert col["manager"] == {"fake": str(root / "r1/grpo/final")} and col["round"] == 2
    sft = json.loads((root / "r2/dynamic/sft/run_signature.json").read_text())
    assert sft["init_adapter"] == str(root / "r1/grpo/final")
    grpo2 = json.loads((root / "r2/dynamic/grpo/training_run.json").read_text())
    assert grpo2["sft_checkpoint"] == str(root / "r2/dynamic/sft/model") and grpo2["config"]["anchor_context"] == "paper"
    # Round 2: the scripted GRPO adapter loses accuracy -> rejected -> G_2 := S_2.
    g2 = json.loads((root / "r2/dynamic/grpo_dev/decision.json").read_text())
    assert not g2["accept"]["accepted"] and g2["checkpoint"] == str(root / "r2/dynamic/sft/model")
    # Parity is reported (subset, not gated); SFT flags are recorded.
    assert json.loads((root / "r1/S1_dev/parity.json").read_text())["status"] == "subset"
    assert "flags" in json.loads((root / "r2/dynamic/sft_dev/decision.json").read_text())
    assert json.loads((root / "prefetch/advisors/prefetch.json").read_text())["counts"]["dev:verifier@S_1"]["requested"] == 4
    # Report: per-round dev metrics, drift on the same ids, Tables 1-2 analogues, label mix, call gap, correction/corruption.
    rep = json.loads((root / "report.json").read_text())
    assert [d["stage"] for d in rep["dev"]] == ["r1/S1_dev", "r1/grpo_dev", "r2/dynamic/sft_dev", "r2/dynamic/grpo_dev"]
    s1 = rep["dev"][0]
    assert s1["accuracy"] == 1.0 and s1["initial_draft_accuracy"] == 0.5 and s1["call_gap"] == 1.0
    assert s1["correction_rate"] == 0.5 and s1["corruption_rate"] == 0.0 and s1["margin"] == 0.25
    bad = rep["dev"][3]
    assert bad["accuracy"] == 0.5 and bad["drift_vs_S1"]["final_regressed"] == 2 and bad["drift_vs_S1"]["calls_changed"] == 2
    assert rep["dev"][1]["drift_vs_S1"] == {"comparable": True, "n": 4, "candidate_new_correct": 0, "candidate_regressed": 0,
                                           "final_new_correct": 0, "final_regressed": 0, "calls_changed": 0,
                                           "call_decision_changed": 0}
    (c,) = rep["collections"]
    assert c["n"] == 8 and {"commit_A0", "best_one_call_A1", "best_measured_AD", "gain_at_1", "n_unsolved"} <= set(c)
    assert set(c["net_marginal_pp"]) == {"extractor", "reasoner", "verifier"} and c["policy_call_rate"] is not None
    assert c["commit_A0"] <= c["best_one_call_A1"] <= c["best_measured_AD"]
    assert rep["labels"][0]["rows"] > 0 and set(rep["labels"][0]["decision_types"]) == {"commit", "call", "commit_after_call"}
    assert [g["stage"] for g in rep["grpo"]] == ["r1/grpo", "r2/dynamic/grpo"]
    assert [t["grpo_rejected"] for t in rep["arms_timeline"]["dynamic"]] == [False, True]
    # Round-over-round drift: S_2 vs G_1, G_2 vs S_2 (same dev ids).
    assert rep["dev"][2]["drift_vs_previous"]["reference"] == "r1/grpo_dev"
    assert rep["dev"][3]["drift_vs_previous"]["reference"] == "r2/dynamic/sft_dev"
    assert rep["dev"][3]["drift_vs_previous"]["final_regressed"] == 2
    # The concurrent prefetch is rechecked sequentially and reported with the S_1 parity.
    recheck = json.loads((root / "prefetch/advisors/prefetch.json").read_text())["sequential_recheck"]
    assert set(recheck["kinds"]) == {"extractor", "reasoner", "verifier"}
    assert all(k["n"] > 0 and k["exact_rate"] == 1.0 for k in recheck["kinds"].values())
    assert json.loads((root / "r1/S1_dev/parity.json").read_text())["advisor_sequential_recheck"] == recheck
    assert "grpo_rejected" in (root / "report.md").read_text() and (root / "dev_metrics.csv").is_file()
    assert not rep["test_sets_used"]
    # A restart re-runs nothing.
    monkeypatch.setattr(CT, "STAGE_FUNCS", {k: (lambda *a, **k: pytest.fail("re-ran")) for k in CT.STAGE_FUNCS})
    assert CT.run(cfg, root, hours=72, executor="inprocess", rt=rt)["complete"]
    monkeypatch.undo()
    # Locked test: S_1 and the pre-registered final of the arm (G_2 = S_2, since GRPO_2 was rejected), once.
    from src.pipeline import stages
    script_tok = ScriptTok(tiny_tokenizer())
    monkeypatch.setattr(stages, "_load_manager_for_eval", lambda ctx, manager_dir, device, dtype:
                        (script_tok, ScriptModel(script_tok, manager_dir)))
    final = CT.final_test(root, executor="inprocess", rt=rt)
    finals = json.loads((root / "final/finals.json").read_text())
    assert finals["locked_test"]["run_dir"] == str(root.resolve())
    assert finals["finals"]["dynamic"]["checkpoint"] == str(root / "r2/dynamic/sft/model") and not finals["incomplete"]
    assert {f["stage"] for f in final["finals"]} == {"final/S_1/test", "final/S_1/dev_forced_verifier",
                                                     "final/dynamic/test", "final/dynamic/dev_forced_verifier"}
    assert final["test_sets_used"]
    s1_test = json.loads((root / "final/S_1/test/final.json").read_text())
    assert s1_test["parity"]["status"] == "not_applicable"  # MedQA's paper targets are dev targets
    monkeypatch.setattr(CT, "STAGE_FUNCS", {k: (lambda *a, **k: pytest.fail("test re-ran")) for k in CT.STAGE_FUNCS})
    CT.final_test(root, executor="inprocess", rt=rt)  # each test set runs once
    assert time.monotonic() - started < 120


def test_status_and_cli_commands(tmp_path, capsys):
    from src.manager.mcq_rsi import __main__ as cli
    cfgfile = tmp_path / "cfg.json"
    cfgfile.write_text(json.dumps({"bench": "medqa", "import_dir": str(tmp_path / "import"),
                                   "advisor_cache": str(tmp_path / "cache")}))
    assert cli.main(["run", "--config", str(cfgfile), "--run-dir", str(tmp_path / "r"), "--dry-run",
                     "--phase", "pilot"]) == 0
    assert "r1/grpo_select" in capsys.readouterr().out and not (tmp_path / "r").exists()
    with pytest.raises(SystemExit):
        cli.main(["run", "--config", str(cfgfile), "--run-dir", str(tmp_path / "r"), "--bench", "gpqa", "--dry-run"])
    run, root = CT.prepare_run(CT.load_config(cfgfile), tmp_path / "r2")
    CT.start(run, root)
    assert cli.main(["status", "--run-dir", str(root)]) == 0
    out = capsys.readouterr().out
    assert "pending  r1/S1_dev" in out
    s = CT.status(root)
    assert len(s["stages"]) == len(run["plan"]["stages"]) and all(r["state"] == "pending" for r in s["stages"])
    assert cli.main(["report", "--run-dir", str(root)]) == 0
    assert "Pending stages" in capsys.readouterr().out


# ------------------------------------------------------------------- review fixes (final PR)

def test_reimport_keeps_the_run_identity(tmp_path):
    """A second ``import`` rewrites every manifest entry's status (downloaded/extracted -> present); runs bind
    to the imported content, so resume and final-test keep working; a changed digest is still refused."""
    from test_mcq_rsi_importer import _fake_bench
    from src.manager.mcq_rsi import importer
    fake, downloader, _ = _fake_bench(tmp_path)
    imp = tmp_path / "import"
    first = importer.run_import(fake, str(imp), downloader=downloader)
    assert {f["status"] for f in first["files"]} == {"downloaded", "extracted"}
    cfg = cfg_for(tmp_path)
    run, root = CT.prepare_run(cfg, tmp_path / "run")
    CT.start(run, root)
    manifest = imp / "medqa" / "import_manifest.json"
    before = manifest.read_bytes()
    assert run["signature"]["manifests"]["import_content_sha256"]
    again = importer.run_import(fake, str(imp), downloader=downloader)
    assert {f["status"] for f in again["files"]} == {"present"} and manifest.read_bytes() != before
    CT.start(CT.prepare_run(cfg, tmp_path / "run")[0], root)  # resume accepted
    CT.check_code_unchanged(json.loads((root / CT.RUN_FILE).read_text()))  # stage subprocesses and final-test too
    m = json.loads(manifest.read_text())
    m["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="code/manifests changed"):
        CT.check_code_unchanged(json.loads((root / CT.RUN_FILE).read_text()))


def _gate_eval(seen, gates):
    from src.manager.mcq_rsi import evaluate as E

    def fake_eval(checkpoint, rows, pool, out, require_gate=True, **kw):
        seen[Path(out).name] = require_gate
        gate = list(gates.get(Path(out).name, []))
        res = {"metrics": _metrics(0.81, 0.42, 0.40), "gate": gate, "passed": not gate}
        Path(out, "mcq_rsi_eval.json").write_text(json.dumps(res))
        if gate and require_gate:
            raise E.EvalGateFailed(f"eval gate failed: {gate}", res)
        return res
    return fake_eval


def test_grpo_candidate_failing_its_eval_gate_is_rejected_not_fatal(tmp_path, monkeypatch):
    from src.manager.mcq_rsi import evaluate as E
    cfg = cfg_for(tmp_path)
    rt = StubRuntime(cfg)
    root = tmp_path / "run"
    s1 = str(Path(cfg["import_dir"]) / "S_1")
    _eval_dir(root, "r1/S1_dev", _metrics(0.80, 0.40, 0.40), {"checkpoint": s1, "role": "s1"})
    for t in ("2e-06", "1e-05"):
        (root / f"r1/grpo_lr{t}").mkdir(parents=True)
        (root / f"r1/grpo_lr{t}/summary.json").write_text(json.dumps({"informativeness": {"passed": True}, "steps": 4,
                                                                      "selected_step": 4, "accepted_by_guard": True}))
    seen = {}
    gates = {"grpo_lr1e-05_dev": ["valid_answer_rate=0.995"], "grpo_lrX_dev": ["advisor failures=2"],
             "sft_dev": ["malformed_tool_calls=1"]}
    monkeypatch.setattr(E, "evaluate", _gate_eval(seen, gates))
    monkeypatch.setattr(StubRuntime, "rows", lambda self, pool: [])
    monkeypatch.setattr(StubRuntime, "pool", lambda self: None)

    def cand(t):
        spec = {"name": f"r1/grpo_lr{t}_dev", "kind": "eval", "lane": "inference",
                "params": {"checkpoint": f"out:r1/grpo_lr{t}::final", "pool": "dev", "role": "grpo",
                           "baseline": "r1/S1_dev", "grpo_stage": f"r1/grpo_lr{t}", "fallback": "import:S_1"}}
        dec = CT.run_stage(root, {"config_full": cfg}, spec, rt)
        CT.validate_stage(spec, root / spec["name"])  # complete although its own gate failed
        return dec

    bad = cand("1e-05")
    assert seen["grpo_lr1e-05_dev"] is False  # the controller asks for the result, not the raise
    assert not bad["accept"]["accepted"] and bad["accept"]["decision"] == "grpo_rejected"
    assert "G_k dev eval failed its gate" in bad["accept"]["reasons"][0]
    assert bad["accept"]["candidate_gate"] == ["valid_answer_rate=0.995"]
    assert bad["checkpoint"] == s1 and bad["dev_result"].endswith("r1/S1_dev/mcq_rsi_eval.json")
    good = cand("2e-06")
    assert good["accept"]["accepted"]
    # The pilot selection skips the gate-failed candidate.
    spec = {"name": "r1/grpo_select", "kind": "grpo_select",
            "params": {"candidates": [{"learning_rate": 2e-6, "grpo": "r1/grpo_lr2e-06", "dev": "r1/grpo_lr2e-06_dev"},
                                      {"learning_rate": 1e-5, "grpo": "r1/grpo_lr1e-05", "dev": "r1/grpo_lr1e-05_dev"}],
                       "baseline": "r1/S1_dev", "default_learning_rate": None}}
    (root / spec["name"]).mkdir(parents=True)
    sel = CT.stage_grpo_select(root, rt, spec, root / spec["name"])
    assert sel["learning_rate"] == 2e-6 and sel["selected"] == "r1/grpo_lr2e-06"
    # Advisor failures stay a hard failure, also for a GRPO candidate.
    (root / "r1/grpo_lrX").mkdir()
    (root / "r1/grpo_lrX/summary.json").write_text((root / "r1/grpo_lr2e-06/summary.json").read_text())
    spec = {"name": "r1/grpo_lrX_dev", "kind": "eval", "params": {"checkpoint": "out:r1/grpo_lrX::final", "pool": "dev",
                                                                   "role": "grpo", "baseline": "r1/S1_dev",
                                                                   "grpo_stage": "r1/grpo_lrX", "fallback": "import:S_1"}}
    (root / spec["name"]).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="advisor infrastructure"):
        CT.stage_eval(root, rt, spec, root / spec["name"])
    assert not (root / spec["name"] / "decision.json").exists()
    # Any other eval (here SFT) still stops on its gate.
    spec = {"name": "r2/dynamic/sft_dev", "kind": "eval", "params": {"checkpoint": "out:r2/dynamic/sft::model",
                                                                      "pool": "dev", "role": "sft",
                                                                      "previous": "decision:r1/grpo_lr2e-06_dev"}}
    (root / spec["name"]).mkdir(parents=True)
    with pytest.raises(E.EvalGateFailed):
        CT.stage_eval(root, rt, spec, root / spec["name"])
    assert seen["sft_dev"] is True


def test_pilot_sweep_trains_each_lr_and_round2_uses_the_selected_lr(tmp_path, monkeypatch):
    from src.manager.mcq_rsi import grpo as G
    cfg = cfg_for(tmp_path, grpo={"learning_rate": 5e-6})  # as in the configs
    rt = StubRuntime(cfg)
    root = tmp_path / "run"
    by = {s["name"]: s for s in CT.build_plan(cfg, "pilot")["stages"]}
    captured = {}

    def fake_train(config, checkpoint, rows, anchor, pool, out, resume=True):
        captured[str(out)] = {**config, "checkpoint": checkpoint, "anchor_rows": anchor}
        return {"steps": 1, "selected_step": 1, "informativeness": {"passed": True}}

    monkeypatch.setattr(G, "train_fa_grpo", fake_train)
    monkeypatch.setattr(G, "recorded_sft_context", lambda checkpoint: None)
    monkeypatch.setattr(StubRuntime, "rows", lambda self, pool: [])
    monkeypatch.setattr(StubRuntime, "pool", lambda self: None)
    monkeypatch.setattr(StubRuntime, "base", lambda self: "BASE")
    Path(cfg["import_dir"]).mkdir(parents=True)
    (Path(cfg["import_dir"]) / "labels.jsonl").write_text(json.dumps({"labels": "round 1"}) + "\n")
    tags = ("2e-06", "5e-06", "1e-05")
    for t in tags:
        CT.run_stage(root, {"config_full": cfg}, by[f"r1/grpo_lr{t}"], rt)
    assert [captured[str(root / f"r1/grpo_lr{t}")]["learning_rate"] for t in tags] == [2e-6, 5e-6, 1e-5]
    (root / "r1/grpo_select").mkdir(parents=True)
    (root / "r1/grpo_select/decision.json").write_text(json.dumps({"checkpoint": "G1@1e-5", "learning_rate": 1e-5}))
    (root / "r2/dynamic/select").mkdir(parents=True)
    (root / "r2/dynamic/select/labels.jsonl").write_text(json.dumps({"labels": "round 2"}) + "\n")
    CT.run_stage(root, {"config_full": cfg}, by["r2/dynamic/grpo"], rt)
    # GRPO_k anchors on round k's labels, round 1 on the imported ones.
    assert captured[str(root / "r1/grpo_lr2e-06")]["anchor_rows"] == [{"labels": "round 1"}]
    assert captured[str(root / "r2/dynamic/grpo")]["anchor_rows"] == [{"labels": "round 2"}]
    assert captured[str(root / "r2/dynamic/grpo")]["learning_rate"] == 1e-5
    assert captured[str(root / "r2/dynamic/grpo")]["checkpoint"] == str(root / "r2/dynamic/sft/model")
    assert json.loads((root / "r2/dynamic/grpo/controller_stage.json").read_text())["learning_rate"] == 1e-5
    # Main phase: the config lr.
    main = {s["name"]: s for s in CT.build_plan(cfg, "main")["stages"]}
    CT.run_stage(tmp_path / "main", {"config_full": cfg}, main["r1/grpo"], rt)
    assert captured[str(tmp_path / "main" / "r1/grpo")]["learning_rate"] == 5e-6


def test_round1_parity_gate_raises_at_stage_level_until_acknowledged(tmp_path, monkeypatch):
    from src.manager.mcq_rsi import evaluate as E
    cfg = cfg_for(tmp_path)
    rt = StubRuntime(cfg)
    root = tmp_path / "run"

    def fake_eval(checkpoint, rows, pool, out, **kw):  # full dev (no limit), 10 pt below the MedQA target
        res = {"metrics": _metrics(0.73, 0.40, 0.30), "gate": [], "passed": True}
        Path(out, "mcq_rsi_eval.json").write_text(json.dumps(res))
        return res

    monkeypatch.setattr(E, "evaluate", fake_eval)
    monkeypatch.setattr(StubRuntime, "rows", lambda self, pool: [])
    monkeypatch.setattr(StubRuntime, "pool", lambda self: None)
    spec = {"name": "r1/S1_dev", "kind": "eval", "params": {"checkpoint": "import:S_1", "pool": "dev", "role": "s1"}}
    out = root / spec["name"]
    out.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="parity gate failed"):
        CT.stage_eval(root, rt, spec, out)
    assert not (out / "decision.json").exists()
    assert json.loads((out / "parity.json").read_text())["status"] == "fail"
    CT.ack_gate(root, "r1/S1_dev", "known 10 pt shift, accepted by the operator")
    dec = CT.stage_eval(root, rt, spec, out)
    assert dec["parity"] == "fail"
    assert json.loads((out / "parity.json").read_text())["acknowledged"]["reason"].startswith("known 10 pt")
    # parity.enforce = false reports without stopping.
    loose = CT.load_config({**{k: cfg[k] for k in ("bench", "import_dir", "advisor_cache")}, "advisor_url": None,
                            "preflight": {"required": False}, "parity": {"enforce": False}})
    (tmp_path / "r2" / "r1" / "S1_dev").mkdir(parents=True)
    assert CT.stage_eval(tmp_path / "r2", StubRuntime(loose), spec, tmp_path / "r2" / "r1" / "S1_dev")["parity"] == "fail"


def _complete(run, root, checkpoint_of=lambda stage: None):
    for s in run["plan"]["stages"]:
        d = Path(root) / s["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / CT.MARKER).write_text("{}")
    for arm, rounds in run["plan"]["by_arm"].items():
        for k, st in rounds.items():
            stage = st["G"][len("decision:"):].partition("#")[0]
            (Path(root) / stage / "decision.json").write_text(json.dumps({"checkpoint": str(Path(root) / stage / "g")}))
    CT._write_json(Path(root) / "budget.json", {"hours": 1.0, "started_unix": time.time(),
                                                "deadline_unix": time.time() + 3600})


def test_controller_guards_failed_evals_changed_markers_retargeting_and_test_retries(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path)
    rt = StubRuntime(cfg)
    run, root = CT.prepare_run(cfg, tmp_path / "run", arms=["dynamic"], rounds=1)
    CT.start(run, root)
    spec = next(s for s in run["plan"]["stages"] if s["name"] == "r1/S1_dev")
    d = _eval_dir(root, "r1/S1_dev", _metrics(0.80, 0.4, 0.4), {"checkpoint": "S1", "role": "s1"})
    CT.validate_stage(spec, d)
    (d / "mcq_rsi_eval.json").write_text(json.dumps({"metrics": {}, "gate": ["malformed_tool_calls=1"], "passed": False}))
    with pytest.raises(RuntimeError, match="eval gate failed"):
        CT.validate_stage(spec, d)
    grpo_spec = next(s for s in run["plan"]["stages"] if s["name"] == "r1/grpo_dev")
    g = _eval_dir(root, "r1/grpo_dev", _metrics(0.80, 0.4, 0.4), {"checkpoint": "G1", "accept": {"accepted": True}})
    (g / "mcq_rsi_eval.json").write_text(json.dumps({"metrics": {}, "gate": ["valid_answer_rate=0.9"], "passed": False}))
    with pytest.raises(RuntimeError, match="eval gate failed"):  # only a rejecting decision completes it
        CT.validate_stage(grpo_spec, g)
    # A completion marker of another spec is refused.
    _eval_dir(root, "r1/S1_dev", _metrics(0.80, 0.4, 0.4), {"checkpoint": "S1", "role": "s1"})
    (d / CT.MARKER).write_text(json.dumps({"spec_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="different spec"):
        CT.execute(root, run, spec, time.time() + 60, executor="inprocess", rt=rt)
    # The pre-registered finals are never re-targeted (no stage may run if the guard regresses).
    def no_stage(*a, **k):
        raise AssertionError("a final stage ran")

    monkeypatch.setitem(CT.STAGE_FUNCS, "test", no_stage)
    monkeypatch.setitem(CT.STAGE_FUNCS, "eval", no_stage)
    _complete(run, root)
    registered = CT.final_test(root, rt=rt, dry_run=True)
    registered["finals"]["dynamic"]["checkpoint"] = "/elsewhere"
    CT._write_json(root / "final" / "finals.json", registered)
    with pytest.raises(RuntimeError, match="never re-targeted"):
        CT.final_test(root, rt=rt, executor="inprocess")
    # retry-stage refuses a locked-test stage, but a forced dev eval of a final may be retried.
    (root / "final" / "finals.json").unlink()
    registered = CT.final_test(root, rt=rt, dry_run=True)
    CT._write_json(root / "final" / "finals.json", registered)
    (root / "final/S_1/test").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="never retried"):
        CT.retry_stage(root, "final/S_1/test")
    (root / "final/S_1/dev_forced_verifier").mkdir(parents=True)
    assert CT.retry_stage(root, "final/S_1/dev_forced_verifier").name.startswith("dev_forced_verifier.failed-")


def test_locked_test_runs_once_per_benchmark_and_never_on_a_pilot(tmp_path, monkeypatch):
    cfg = cfg_for(tmp_path, final={"test_pools": ["test"], "forced": ["verifier"]})
    rt = StubRuntime(cfg)
    ran = []

    def fake_test(root, rt, spec, out):
        ran.append(str(out))
        CT._write_json(out / "final.json", {"metrics": {"n": 1, "accuracy": 1.0, "calls_per_example": 0.0}})
        return {}

    monkeypatch.setitem(CT.STAGE_FUNCS, "test", fake_test)
    monkeypatch.setattr(CT, "report", lambda root: {"finals": []})

    def finished(name, phase="main"):
        run, root = CT.prepare_run(cfg, tmp_path / name, phase=phase, arms=["dynamic"], rounds=1)
        CT.start(run, root)
        _complete(run, root)
        return root

    a = finished("a")
    real = CT.code_identity

    def changed():
        ident = real()
        ident["files"]["manager/mcq_rsi/evaluate.py"] = "0" * 64
        return ident

    monkeypatch.setattr(CT, "code_identity", changed)  # changed code: refused before anything is registered
    with pytest.raises(RuntimeError, match="code/manifests changed"):
        CT.final_test(a, rt=rt, executor="inprocess")
    assert not CT.locked_test_registry(cfg).exists() and not (a / "final").exists() and not ran
    monkeypatch.setattr(CT, "code_identity", real)
    CT.final_test(a, rt=rt, executor="inprocess")
    reg = json.loads(CT.locked_test_registry(cfg).read_text())
    assert [r["run_dir"] for r in reg["registrations"]] == [str(a)] and reg["registrations"][0]["test_pools"] == ["test"]
    n = len(ran)
    CT.final_test(a, rt=rt, executor="inprocess")  # the same directory resumes; nothing reruns
    assert len(ran) == n
    b = finished("b")
    with pytest.raises(RuntimeError, match="already registered"):
        CT.final_test(b, rt=rt, executor="inprocess")
    assert not (b / "final" / "finals.json").exists()
    CT.final_test(b, rt=rt, executor="inprocess", reuse_test="pod lost the first run's checkpoints")
    reg = json.loads(CT.locked_test_registry(cfg).read_text())
    assert [r["run_dir"] for r in reg["registrations"]] == [str(a), str(b)]
    assert reg["registrations"][1]["reuse_reason"] == "pod lost the first run's checkpoints"
    assert json.loads((b / "final" / "finals.json").read_text())["locked_test"]["previous_registrations"] == 1
    p = finished("p", phase="pilot")
    with pytest.raises(RuntimeError, match="pilot run never runs the locked test"):
        CT.final_test(p, rt=rt, executor="inprocess")
    assert CT.final_test(p, rt=rt, dry_run=True)["finals"]  # a dry run only shows them


def test_pilot_report_marks_g1_rejected_when_no_lr_was_accepted(tmp_path):
    cfg = cfg_for(tmp_path)
    run, root = CT.prepare_run(cfg, tmp_path / "run", phase="pilot", arms=["dynamic"], rounds=1)
    CT.start(run, root)
    d = root / "r1/grpo_select"
    d.mkdir(parents=True)
    (d / "decision.json").write_text(json.dumps({"checkpoint": "S1", "learning_rate": 5e-6, "selected": None,
                                                 "role": "grpo_select", "candidates": [],
                                                 "metrics": _metrics(0.8, 0.4, 0.3)}))
    (d / CT.MARKER).write_text("{}")
    rep = CT.report(root)
    assert rep["arms_timeline"]["dynamic"] == [{"round": 1, "G_stage": "r1/grpo_select", "checkpoint": "S1",
                                                "grpo_rejected": True, "accuracy": 0.8, "calls_per_example": 0.4,
                                                "call_gap": 0.3}]


_STAGE = ("import os, time  # src.manager.mcq_rsi stage (test)\nprint(os.getpid(), flush=True)\ntime.sleep(120)")


def test_sigterm_or_sighup_kills_the_stage_group_and_restarts_refuse_live_orphans(tmp_path):
    script = tmp_path / "controller.py"
    script.write_text(
        "import os, sys, time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from src.manager.mcq_rsi import controller as CT\n"
        f"st = CT.Status({str(tmp_path)!r}, 1, time.time() + 600, heartbeat=0.1)\n"
        "with CT.stop_on_signals():\n"
        "    try:\n"
        f"        CT.run_subprocess([sys.executable, '-c', {_STAGE!r}], Path({str(tmp_path / 'stage.log')!r}),\n"
        "                          dict(os.environ), time.time() + 600, st, poll=0.1)\n"
        "    except CT.ControllerSignal:\n"
        "        sys.exit(7)\n")
    status = tmp_path / "status.json"
    for sig in (signal.SIGTERM, signal.SIGHUP):
        status.unlink(missing_ok=True)
        proc = subprocess.Popen([sys.executable, str(script)], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            pid, deadline = None, time.time() + 60
            while pid is None and time.time() < deadline:
                with_pid = json.loads(status.read_text()) if status.is_file() else {}
                pid = with_pid.get("stage_pid")
                time.sleep(0.1)
            assert pid and CT.live_stage_process(pid)
            proc.send_signal(sig)
            assert proc.wait(timeout=60) == 7, proc.stderr.read()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stderr.close()
        time.sleep(0.3)
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)  # the stage's whole process group is gone
        assert json.loads(status.read_text())["stage_pid"] is None and CT.live_stage_process(pid) is None
    # A live stage group of a dead controller blocks a restart; a reused pid running something else does not.
    orphan = subprocess.Popen([sys.executable, "-c", _STAGE], start_new_session=True, stdout=subprocess.DEVNULL)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        cfg = cfg_for(tmp_path)
        run, root = CT.prepare_run(cfg, tmp_path / "run")
        CT.start(run, root)
        CT._write_json(root / "status.json", {"stage_pid": orphan.pid, "controller": "running"})
        with pytest.raises(RuntimeError, match="still running"):
            CT.run(cfg, root, hours=1, executor="inprocess", rt=StubRuntime(cfg))
        with pytest.raises(RuntimeError, match="still running"):
            CT.final_test(root, rt=StubRuntime(cfg), accept_incomplete="x")
        assert CT.status(root)["stage_process_alive"]
        CT._write_json(root / "status.json", {"stage_pid": other.pid, "controller": "running"})
        CT.refuse_orphaned_stage(root)
    finally:
        for p in (orphan, other):
            p.kill()
            p.wait()
    # A stage directory is executed by one process at a time; the lock file stays outside the directory.
    out = tmp_path / "run" / "r1" / "grpo"
    with CT.stage_lock(out):
        with pytest.raises(RuntimeError, match="another process"):
            with CT.stage_lock(out):
                pass
    assert not any(out.iterdir()) and CT.stage_lock_path(out).name == ".grpo.stage.lock"


# ----------------------------------------------------------------------------------- scripts

@pytest.mark.parametrize("script", ["start_mcq_advisors.sh", "setup_mcq_rsi_pod.sh", "runpod_mcq_rsi.sh"])
def test_bash_scripts_parse(script):
    path = ROOT / "scripts" / script
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True, timeout=30).returncode == 0
    text = path.read_text()
    assert text.startswith("#!/usr/bin/env bash") and "set -euo pipefail" in text
    assert not re.search(r"(hf_[A-Za-z0-9]{20,}|WANDB_API_KEY=|HF_TOKEN=)", text)  # never store tokens
    shellcheck = subprocess.run(["which", "shellcheck"], capture_output=True, text=True).stdout.strip()
    if shellcheck:
        assert subprocess.run([shellcheck, "-S", "error", str(path)], capture_output=True, timeout=60).returncode == 0


def test_advisor_script_serving_flags():
    text = (ROOT / "scripts" / "start_mcq_advisors.sh").read_text()
    for flag in ("--enable-lora", "--max-lora-rank 16", "--max-loras", "--max-model-len", "--gpu-memory-utilization",
                 "--served-model-name", "--revision", "--lora-modules"):
        assert flag in text, flag
    assert 'PORT="${MCQ_ADVISOR_PORT:-18002}"' in text and 'GPU_UTIL="${MCQ_ADVISOR_GPU_UTIL:-0.45}"' in text
    assert "port_free" in text and "8001" in text and 'VLLM_VENV="${VLLM_VENV:-/workspace/vllm-venv}"' in text
    assert "serve-loras" in text and 'LORA_MODE="${LORA_MODE:-multimodal}"' in text
    assert "c202236235762e1c871ad0ccb60c8ee5ba337b9a" in text and "hf-overrides" not in text


def _bash_function(text, name):
    return text.split(f"{name}() {{", 1)[1].split("\n}\n", 1)[0]


def test_pod_scripts_and_runbook_operations():
    wrapper = (ROOT / "scripts" / "runpod_mcq_rsi.sh").read_text()
    pipeline = _bash_function(wrapper, "pipeline")
    assert "step_smoke_check" in pipeline and "step_pilot" not in pipeline  # the pilot is started explicitly
    assert "smoke_passed" in _bash_function(wrapper, "step_pilot")
    assert "PREFLIGHT_BENCHES:-$BENCH" in wrapper and "--skip-if-passed" in wrapper
    assert 'SMOKE_RUN="${SMOKE_RUN:-$WORK/runs/smoke}"' in wrapper and "RUN_DIR:-" in wrapper and "--reuse-test" in wrapper
    assert "MARGENT_WANDB_MODE:-disabled" not in wrapper  # W&B follows the config unless the operator disables it
    setup = (ROOT / "scripts" / "setup_mcq_rsi_pod.sh").read_text()
    assert "backups,recorded}" in setup and "hf auth login" in setup and "huggingface-cli" not in setup
    assert "MCQ_RSI_TOKENIZER_DIR" in setup and "faulthandler_timeout=60" in setup and "timeout 900" in setup
    assert 'out="$(CUDA_VISIBLE_DEVICES= ' in setup  # CPU unit tests stay on the CPU on a GPU pod
    assert "VLLM_TORCH_INDEX_URL" in setup and ".mcq_build" in setup
    assert setup.index("cuda_major=") < setup.index("torch in the vLLM venv sees no GPU")  # specific message first
    advisors = (ROOT / "scripts" / "start_mcq_advisors.sh").read_text()
    assert "vllm.entrypoints" in _bash_function(advisors, "served_pid") and "server_flags_${PORT}.json" in advisors
    assert '>> "$LOG"' not in advisors  # a fresh log per start
    assert 'MCQ_PYTHON="$PY"' in wrapper and 'MCQ_WORK="$WORK"' in wrapper and 'TMPDIR="$TMPDIR"' in wrapper  # bg env
    assert 'cli report --run-dir "${RUN_DIR:-' in wrapper
    book = (ROOT / "docs" / "MCQ_RSI_RUNBOOK.md").read_text()
    assert "export PY=/workspace/mcq-venv/bin/python HF_HOME=/workspace/hf-cache" in book
    assert "git checkout feat/mcq-rsi-controller" in book and "huggingface-cli login" not in book
    assert "bg main" in book and "bg final-test" in book and "--reuse-test" in book and "SM 12.x" in book


def test_advisor_script_stale_pid_file_is_removed_and_never_returned_as_a_pid(tmp_path):
    advisors = (ROOT / "scripts" / "start_mcq_advisors.sh").read_text()
    pidfile = tmp_path / "advisors.pid"
    pidfile.write_text(str(os.getpid()))  # alive, but not a vLLM server
    script = ("set -euo pipefail\nlog() { printf '[mcq-advisors] %s\\n' \"$*\"; }\n"
              f"PIDFILE={pidfile}\nPORT=18002\nserved_pid() {{{_bash_function(advisors, 'served_pid')}\n}}\n"
              'pid="$(served_pid)"\nprintf "pid=[%s]" "$pid"\n')
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout == "pid=[]" and "stale pid file" in out.stderr and not pidfile.exists()


def test_wandb_logs_dev_and_grpo_step_curves_and_never_fails_the_run(tmp_path, monkeypatch, capsys):
    import types
    logged, inits, summaries = [], [], {}

    class Table:
        def __init__(self, columns, data):
            self.columns, self.data = columns, data

    class Run:
        summary = types.SimpleNamespace(update=summaries.update)

        def log(self, d):
            logged.append(d)

        def finish(self):
            pass

    fake = types.SimpleNamespace(init=lambda **kw: inits.append(kw) or Run(), Table=Table,
                                 plot=types.SimpleNamespace(line=lambda t, x, y, title: ("line", x, y, len(t.data))))
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("MARGENT_WANDB_MODE", raising=False)
    g = tmp_path / "r1/grpo"
    g.mkdir(parents=True)
    steps = [{"step": 1, "loss": 0.5, "kl_dec": 0.0, "kl_root": 0.0, "train_call_rate": 0.4, "J_mean": 0.6,
              "n_states": 10, "n_informative": 3, "learning_rate": 5e-6},
             {"step": 2, "loss": 0.4, "kl_dec": 0.01, "kl_root": 0.002, "train_call_rate": 0.42, "J_mean": 0.62,
              "n_states": 10, "n_informative": 4, "guard": {"call_rate": 0.41, "kl_dec": 0.01, "passed": True}}]
    (g / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steps))
    result = {"bench": "medqa", "phase": "pilot", "arms": ["dynamic"], "rounds": 2, "completed_stages": 3,
              "planned_stages": 9, "dev": [{"stage": "r1/S1_dev", "round": 1, "role": "s1", "n": 200, "accuracy": 0.8,
                                            "calls_per_example": 0.4, "call_gap": 0.2}],
              "grpo": [{"stage": "r1/grpo", "selected_step": 2, "informative_fraction": 0.35, "learning_rate": 5e-6}]}
    cfg = {"wandb": {"enabled": True, "entity": "e", "project": "MCQ_rsi"}}
    CT._wandb_log(cfg, result, "run_abc", tmp_path)
    assert inits[0]["project"] == "MCQ_rsi" and inits[0]["resume"] == "allow" and inits[0]["id"] == "mcq_rsi_medqa_pilot_run_abc"
    log = logged[0]
    assert log["progress/completed_stages"] == 3 and log["dev/table"].data[0][:5] == ["r1/S1_dev", 1, "s1", 200, 0.8]
    table = log["grpo/r1/grpo/steps"]
    assert table.columns[0] == "step" and "guard_call_rate" in table.columns and "guard_passed" not in table.columns
    rows = [dict(zip(table.columns, r)) for r in table.data]
    assert [r["informative_fraction"] for r in rows] == [0.3, 0.4] and rows[0]["guard_call_rate"] is None
    assert log["grpo/r1/grpo/kl_dec"] == ("line", "step", "kl_dec", 2)
    assert summaries["dev/r1/S1_dev/accuracy"] == 0.8 and summaries["grpo/r1/grpo/selected_step"] == 2
    # Disabled by config or env: nothing; a W&B error is printed, never raised.
    logged.clear()
    CT._wandb_log({"wandb": {"enabled": False}}, result, "k", tmp_path)
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    CT._wandb_log(cfg, result, "k", tmp_path)
    assert not logged
    monkeypatch.delenv("MARGENT_WANDB_MODE")
    fake.init = lambda **kw: (_ for _ in ()).throw(RuntimeError("not logged in"))
    CT._wandb_log(cfg, result, "k", tmp_path)
    assert "W&B logging failed" in capsys.readouterr().out
