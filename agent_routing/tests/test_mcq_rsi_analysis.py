"""scripts/mcq_rsi_analysis.py: paired bootstrap, McNemar, matched-budget replay and run tables (design §5)."""
import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mcq_rsi_analysis", ROOT / "scripts" / "mcq_rsi_analysis.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def test_percentile_matches_numpy():
    np = pytest.importorskip("numpy")
    rng = random.Random(3)
    xs = sorted(rng.random() for _ in range(997))
    for q in (0.0, 0.025, 0.5, 0.975, 1.0):
        assert A.percentile(xs, q) == pytest.approx(float(np.percentile(xs, 100 * q)), abs=1e-12)


def test_paired_bootstrap_degenerate_and_deterministic():
    same = A.paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1], n_boot=500)
    assert same["diff"] == 0 and same["ci95"] == [0, 0] and same["p_two_sided"] == 1.0
    up = A.paired_bootstrap([0] * 50, [1] * 50, n_boot=500)
    assert up["diff"] == 1 and up["ci95"] == [1, 1] and up["p_two_sided"] == 0.0
    a, b = [random.Random(1).random() < 0.5 for _ in range(60)], [random.Random(2).random() < 0.5 for _ in range(60)]
    assert A.paired_bootstrap(a, b, 1000, seed=7) == A.paired_bootstrap(a, b, 1000, seed=7)
    with pytest.raises(ValueError):
        A.paired_bootstrap([1], [1, 0])


def test_paired_bootstrap_interval_matches_the_sampling_distribution():
    """n = 400 paired binary outcomes with a true +10 pt shift: the 10k interval is mean +- 1.96 SE."""
    rng = random.Random(0)
    a, b = [], []
    for _ in range(400):
        base = rng.random() < 0.6
        a.append(base)
        b.append(True if (not base and rng.random() < 0.3) else (base and rng.random() > 0.05) or (not base and False))
    diffs = [int(y) - int(x) for x, y in zip(a, b)]
    mean = sum(diffs) / len(diffs)
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / len(diffs) / len(diffs))
    res = A.paired_bootstrap(a, b, n_boot=10_000, seed=1)
    assert res["diff"] == pytest.approx(mean)
    lo, hi = res["ci95"]
    assert lo == pytest.approx(mean - 1.96 * se, abs=0.25 * se) and hi == pytest.approx(mean + 1.96 * se, abs=0.25 * se)
    assert lo > 0 and res["p_two_sided"] < 0.01


def test_mcnemar_exact_and_chi_square():
    a = [True] * 10 + [False] * 2 + [True] * 30 + [False] * 8  # 10 only-a, 2 only-b, 38 concordant
    b = [False] * 10 + [True] * 2 + [True] * 30 + [False] * 8
    m = A.mcnemar(a, b)
    assert (m["only_a"], m["only_b"]) == (10, 2)
    assert m["exact_p"] == pytest.approx(2 * (1 + 12 + 66) / 4096)  # two-sided binomial, n = 12
    assert m["chi2_cc"] == pytest.approx(49 / 12)
    assert m["chi2_p"] == pytest.approx(0.043308, abs=2e-6)  # chi2.sf(49/12, 1)
    assert A.mcnemar(b, a)["exact_p"] == m["exact_p"]
    assert A.mcnemar([True, False], [True, False]) == {"only_a": 0, "only_b": 0, "exact_p": 1.0, "chi2_cc": 0.0, "chi2_p": 1.0}
    assert A.mcnemar([True] * 5, [False] * 5)["exact_p"] == pytest.approx(2 / 32)


def rec(eid, cand, correct, tools=()):
    return {"example_id": eid, "initial_draft_correct": cand, "correct": correct, "tool_calls": len(tools),
            "tool_names_called": [f"{t}_tool" for t in tools], "initial_draft": "A"}


def synthetic(n=200, seed=0):
    rng = random.Random(seed)
    cand = {i: rng.random() < 0.55 for i in range(n)}
    forced_v = {i: rec(i, cand[i], (not cand[i] and rng.random() < 0.6) or (cand[i] and rng.random() > 0.08), ("verifier",))
                for i in range(n)}
    return cand, forced_v


def test_matched_budget_replay_separates_selective_from_random_calling():
    cand, forced_v = synthetic()
    ids = sorted(cand)
    wrong = [i for i in ids if not cand[i]][:40]
    selective = {i: rec(i, cand[i], forced_v[i]["correct"] if i in wrong else cand[i], ("verifier",) if i in wrong else ())
                 for i in ids}
    res = A.matched_budget_replay(selective, {"verifier": forced_v}, n_resamples=2000, seed=3)
    assert res["role_counts"] == {"verifier": 40} and res["fixed_unmatched"] == 0
    assert res["policy_replay_accuracy"] == pytest.approx(sum(selective[i]["correct"] for i in ids) / len(ids))
    assert res["above_random_interval"] and res["p_random_ge_policy"] < 0.01
    # The random baseline's mean is the analytic expectation of 40 uniformly placed Verifier calls.
    gain = sum(forced_v[i]["correct"] - cand[i] for i in ids) / len(ids)
    expected = sum(cand.values()) / len(ids) + 40 * gain / len(ids)
    assert res["random_mean"] == pytest.approx(expected, abs=0.004)
    rng = random.Random(9)
    picked = set(rng.sample(ids, 40))
    random_policy = {i: rec(i, cand[i], forced_v[i]["correct"] if i in picked else cand[i], ("verifier",) if i in picked else ())
                     for i in ids}
    r2 = A.matched_budget_replay(random_policy, {"verifier": forced_v}, n_resamples=2000, seed=3)
    assert r2["random_ci95"][0] <= r2["policy_replay_accuracy"] <= r2["random_ci95"][1]
    # A role without a forced eval is held fixed, not resampled.
    mixed = dict(selective)
    for i in wrong[:5]:
        mixed[i] = rec(i, cand[i], True, ("reasoner",))
    r3 = A.matched_budget_replay(mixed, {"verifier": forced_v}, n_resamples=200)
    assert r3["fixed_unmatched"] == 5 and r3["role_counts"] == {"verifier": 35}
    with pytest.raises(ValueError):
        A.matched_budget_replay(selective, {"verifier": {i: forced_v[i] for i in ids[:10]}})


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_compare_agree_and_run_tables(tmp_path, capsys):
    cand, forced_v = synthetic(100, seed=4)
    ids = sorted(cand)
    s1 = [rec(i, cand[i], cand[i]) for i in ids]
    dyn = [rec(i, cand[i], forced_v[i]["correct"] if not cand[i] else cand[i], ("verifier",) if not cand[i] else ())
           for i in ids]
    root = tmp_path / "run"
    write_jsonl(root / "r1/S1_dev/manager_tool_eval.jsonl", s1)
    (root / "r1/S1_dev/mcq_rsi_eval.json").write_text("{}")
    write_jsonl(root / "r2/dynamic/grpo_dev/manager_tool_eval.jsonl", dyn)
    (root / "r2/dynamic/grpo_dev/decision.json").write_text(json.dumps(
        {"dev_result": str(root / "r2/dynamic/grpo_dev/mcq_rsi_eval.json")}))
    stages = []
    for label, rows in (("S_1", s1), ("dynamic", dyn)):
        write_jsonl(root / f"final/{label}/test/manager_tool_eval.jsonl", rows)
        write_jsonl(root / f"final/{label}/dev_forced_verifier/manager_forced_verifier.jsonl", list(forced_v.values()))
        stages += [{"name": f"final/{label}/test", "params": {"label": label, "pool": "test", "checkpoint": "x"}},
                   {"name": f"final/{label}/dev_forced_verifier",
                    "params": {"label": label, "pool": "dev", "checkpoint": "x", "forced": "verifier"}}]
    (root / "final/finals.json").write_text(json.dumps({"finals": {"S_1": {"ref": "import:S_1"},
                                                                   "dynamic": {"ref": "decision:r2/dynamic/grpo_dev"}},
                                                        "stages": stages}))
    coll = {"stage": "r2/collect", "round": 2, "n": 400, "commit_A0": 0.76, "best_one_call_A1": 0.9175,
            "best_measured_AD": 0.9225, "gain_at_1": 0.969, "n_unsolved": 31,
            "net_marginal_pp": {"extractor": 2.0, "reasoner": 6.8, "verifier": 10.0}}
    dev = {"stage": "r1/S1_dev", "round": 1, "initial_draft_accuracy": 0.685, "margin": 0.435, "call_gap": 0.291,
           "calls_per_example": 1.0, "correction_rate": 0.21, "corruption_rate": 0.035}
    final = {"stage": "final/dynamic/test", "label": "dynamic", "pool": "test", "forced": None, "n": 100, "candidate": 0.5,
             "accuracy": 0.7, "gain_pp": 20.0, "calls_per_example": 0.4}
    (root / "report.json").write_text(json.dumps({"collections": [coll], "dev": [dev], "finals": [final]}))
    result = A.run_tables(root, n_boot=500, n_replay=300)
    t = result["tables"]
    assert t["table1_oracle_by_depth"]["rows"][0][3:7] == pytest.approx([76.0, 91.75, 92.25, 0.969])
    assert t["table2_net_marginal_value"]["rows"][0][2:] == [2.0, 6.8, 10.0]
    assert t["table7_selectivity"]["rows"][0][2:] == [0.685, 0.435, 0.291]
    assert t["table8_outcomes"]["rows"][0][2:] == [1.0, 0.21, 0.035]
    assert t["table3_test"]["rows"][0][1:4] == ["dynamic", "test", 100]
    (comp,) = result["stats"]["comparisons"]
    assert (comp["a"], comp["b"], comp["pool"]) == ("S_1", "dynamic", "test")
    direct = A.compare(A.load_eval(root / "final/S_1/test"), A.load_eval(root / "final/dynamic/test"), 500)
    assert comp["accuracy"] == direct["accuracy"] and comp["mcnemar"]["only_a"] == 0 < comp["mcnemar"]["only_b"]
    assert comp["calls"]["diff"] == pytest.approx(sum(not c for c in cand.values()) / 100)
    replays = {r["label"]: r for r in result["stats"]["replay"]}
    assert replays["dynamic"]["above_random_interval"] and replays["S_1"]["role_counts"] == {"verifier": 0}
    md = A.write_tables(result, tmp_path / "out")
    assert {p.name for p in (tmp_path / "out").iterdir()} >= {"tables.md", "analysis.json", "paired_tests.csv",
                                                               "matched_budget_replay.csv", "table1_oracle_by_depth.csv"}
    assert "Matched-budget replay" in md.read_text()
    assert A.main(["tables", "--run-dir", str(root), "--n-boot", "200", "--n-replay", "100"]) == 0
    assert "table3_test" in capsys.readouterr().out
    assert A.main(["compare", "--a", str(root / "final/S_1/test"), "--b", str(root / "final/dynamic/test"),
                   "--n-boot", "200"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 100
    ag = A.agreement(A.load_eval(root / "final/S_1/test"), A.load_eval(root / "final/dynamic/test"))
    assert ag["n_common"] == 100 and ag["agree"] == sum(cand.values())
    with pytest.raises(ValueError, match="identical ids"):
        A.compare({1: s1[1]}, {2: s1[2]})
    # A forced eval flagged broken (too many invalid answers) is left out of the replay; its role's examples stay fixed.
    (root / "final/dynamic/dev_forced_verifier/final.json").write_text(json.dumps(
        {"broken": True, "metrics": {"valid_answer_rate": 0.63}}))
    replays = {r["label"]: r for r in A.run_tables(root, n_boot=100, n_replay=100)["stats"]["replay"]}
    assert "dynamic" not in replays or replays["dynamic"]["excluded_broken_roles"] == {"verifier": 0.63}
    assert replays["S_1"]["excluded_broken_roles"] == {}


def _rec(i, draft, tools, correct=True):
    return {"example_id": i, "initial_draft": draft, "tool_names_called": tools, "correct": correct,
            "initial_draft_correct": correct, "tool_calls": len(tools)}


def test_smoke_check_gates_round0_agreement_memory_and_accept_blocks(tmp_path, capsys):
    import io
    import tarfile
    run = tmp_path / "smoke"
    (run / "r1" / "S1_dev").mkdir(parents=True)
    ours = [_rec(i, "A", ["verifier_tool"] if i % 3 == 0 else []) for i in range(50)]
    (run / "r1/S1_dev/manager_tool_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ours))
    paper = [dict(r) for r in ours]
    paper[0]["initial_draft"], paper[1]["tool_names_called"] = "B", ["extractor_tool"]  # 48 of 50 agree
    member = "outputs/eval/medqa_9b_d2400_ev_r3/manager_tool_eval.jsonl"
    data = "".join(json.dumps(r) + "\n" for r in paper).encode()
    with tarfile.open(tmp_path / "assets.tgz", "w:gz") as tar:
        info = tarfile.TarInfo("./" + member)  # members may carry a leading ./
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    (run / "r1/grpo").mkdir(parents=True)
    (run / "r1/grpo/controller_stage.json").write_text(json.dumps({"peak_memory_gb": 41.5}))
    report = {"complete": True, "pending": [], "dev": [{"stage": "r1/S1_dev", "gate": []}],
              "decisions": [{"stage": "r1/grpo_dev", "role": "grpo", "accept": {"decision": "grpo_accepted"}}]}
    (run / "report.json").write_text(json.dumps(report))
    argv = ["smoke-check", "--run-dir", str(run), "--recorded", str(tmp_path / "assets.tgz"), "--member", member,
            "--extract-to", str(tmp_path / "recorded")]
    assert A.main(argv) == 0
    out = json.loads((run / "smoke_check.json").read_text())
    assert out["passed"] and (tmp_path / "recorded" / member).is_file()
    agree = next(c for c in out["checks"] if "agreement" in c["check"])
    assert agree["detail"]["agree"] == 48 and agree["detail"]["disagree_ids"] == [0, 1]
    assert any("forced rollback" in m for m in out["manual"]) and "MANUAL" in capsys.readouterr().out
    # 47/50, a peak above 60 GB, a missing accept block or an unfinished run each fail it.
    paper[2]["initial_draft"] = "C"
    (tmp_path / "recorded" / member).write_text("".join(json.dumps(r) + "\n" for r in paper))
    assert A.main(argv) == 1
    assert not next(c for c in json.loads((run / "smoke_check.json").read_text())["checks"] if "agreement" in c["check"])["passed"]
    paper[2]["initial_draft"] = "A"
    (tmp_path / "recorded" / member).write_text("".join(json.dumps(r) + "\n" for r in paper))
    assert A.main(argv) == 0
    (run / "r1/grpo/controller_stage.json").write_text(json.dumps({"peak_memory_gb": 61.0}))
    assert A.main(argv) == 1
    (run / "r1/grpo/controller_stage.json").write_text(json.dumps({"peak_memory_gb": 41.5}))
    (run / "report.json").write_text(json.dumps({**report, "decisions": [{"stage": "r1/grpo_dev", "role": "grpo"}]}))
    assert A.main(argv) == 1
    (run / "report.json").write_text(json.dumps({**report, "complete": False, "pending": ["r2/collect"]}))
    assert A.main(argv) == 1
    (run / "report.json").write_text(json.dumps({**report, "dev": [{"stage": "r1/S1_dev", "gate": ["malformed_tool_calls=1"]}]}))
    assert A.main(argv) == 1
    (run / "report.json").write_text(json.dumps(report))
    assert A.main(argv) == 0
    (run / "r1/grpo/controller_stage.json").write_text(json.dumps({"peak_memory_gb": None}))  # not recorded
    assert A.main(argv) == 1
