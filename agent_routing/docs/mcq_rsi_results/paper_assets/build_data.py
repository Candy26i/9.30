#!/usr/bin/env python3
# NOTE: this script reads the raw result folders from the Claude scratchpad of 2026-10-11 (paths below); in the repo,
# data.json is the frozen output and make_all.sh (without --data) rebuilds every figure and table from it.
"""Consolidate every number the paper figures/tables need into data.json, reading the raw result files wherever they exist
locally and recomputing paired tests from the per-question locked-test records. Numbers that exist only on the results page
(docs/mcq_rsi_results/margent_rsi_results.html) are hard-coded with their provenance. Run from anywhere."""
import csv, json, os, re, subprocess, sys
from pathlib import Path

SCR = Path("/private/tmp/claude-501/-Users-madili-Desktop-7-98/17a9c165-9b16-40c6-bd20-a405349d1016/scratchpad")
RUNS = SCR / "results/runs"; ARCH = SCR / "locked_archive/runs"; TOK = SCR / "results/tokcost"; SA = SCR / "results/single_agent"
REPO = Path("/Users/madili/Desktop/7.98/agent_routing"); ANALYSIS = REPO / "scripts/mcq_rsi_analysis.py"
OUT = SCR / "paper_assets/data.json"
BENCHES = ["medqa", "mmlu_pro", "gpqa", "aqua"]
NAMES = {"medqa": "MedQA", "mmlu_pro": "MMLU-Pro", "gpqa": "GPQA", "aqua": "AQuA-RAT"}
POOLS = {"medqa": ["test"], "mmlu_pro": ["test", "test_paper"], "gpqa": ["test"], "aqua": ["test"]}
N_TEST = {"medqa": 500, "mmlu_pro": 500, "gpqa": 100, "aqua": 254}; N_DEV = {"medqa": 200, "mmlu_pro": 200, "gpqa": 100, "aqua": 254}
RHO = {"medqa": 3.0, "mmlu_pro": 2.0, "gpqa": 2.0, "aqua": 1.0}  # src/manager/mcq_rsi/benchmarks.py

def final_dir(run, arm, pool):
    for base in (RUNS, ARCH):
        d = base / run / "final" / arm / pool
        if (d / "mcq_rsi_eval.json").is_file() or (d / "final.json").is_file():
            return d
    return None

def metrics(run, arm, pool):
    d = final_dir(run, arm, pool)
    if d is None:
        return None
    m = json.load(open(d / ("mcq_rsi_eval.json" if (d / "mcq_rsi_eval.json").is_file() else "final.json")))["metrics"]
    return {"acc": round(m["accuracy"] * 100, 2), "calls": round(m["calls_per_example"], 3), "draft_acc": round(m["initial_draft_accuracy"] * 100, 2),
            "n": m["n"], "valid": m["valid_answer_rate"], "correction": m["correction_rate"], "corruption": m["corruption_rate"],
            "call_rate_draft_wrong": m.get("call_rate_given_draft_wrong"), "src": str(d.relative_to(SCR))}

_cmp_cache = {}
def compare(a, b):
    """Paired bootstrap (10k) + exact McNemar between two final dirs; returns None when either is missing."""
    if a is None or b is None:
        return None
    key = (str(a), str(b))
    if key not in _cmp_cache:
        out = subprocess.run([sys.executable, str(ANALYSIS), "compare", "--a", str(a), "--b", str(b)], capture_output=True, text=True, check=True).stdout
        d = json.loads(out)
        _cmp_cache[key] = {"d_acc": round(d["accuracy"]["diff"] * 100, 2), "p": round(d["accuracy"]["p_two_sided"], 4), "ci": [round(x * 100, 2) for x in d["accuracy"]["ci95"]],
                           "mcnemar_p": round(d["mcnemar"]["exact_p"], 4), "d_calls": round(d["calls"]["diff"], 3), "p_calls": round(d["calls"]["p_two_sided"], 4), "n": d["n"]}
    return _cmp_cache[key]

def v3g_run(b, arm):
    return {"medqa": "medqa_v3g", "gpqa": "gpqa_v3g", "aqua": "aqua_v3g"}.get(b) or {"S_1": "mmlu_pro_v3g_dyn", "dynamic": "mmlu_pro_v3g_dyn", "static": "mmlu_pro_v3g_sta", "success": "mmlu_pro_v3g_suc"}[arm]

data = {"meta": {"benches": BENCHES, "names": NAMES, "pools": POOLS, "n_test": N_TEST, "n_dev": N_DEV,
                 "note": "Accuracies in %, calls per question, tokens per question. Locked tests: MedQA 500, MMLU-Pro 500 (test; test_paper overlaps training pools), GPQA 100 (Diamond), AQuA 254. p = paired bootstrap (10k) two-sided unless noted."}}

# ---------------- 1. locked tests: v3+GRPO (main), v3 no-GRPO (ablation), v3 rounds 4-5 ----------------
locked = {}
for b in BENCHES:
    for pool in POOLS[b]:
        key = f"{b}:{pool}"; locked[key] = {}
        for arm in ("S_1", "dynamic", "static", "success"):
            locked[key][f"v3g/{arm}"] = metrics(v3g_run(b, arm), arm, pool)
        for arm in ("S_1", "dynamic_sft", "static_sft"):
            locked[key][f"v3/{arm}"] = metrics(f"{b}_v3", arm, pool)
        if b != "gpqa":
            for arm in ("dynamic_sft", "static_sft"):
                locked[key][f"v3r5/{arm}"] = metrics(f"{b}_v3r5", arm, pool)
data["locked"] = locked

# paired comparisons
pairs = {}
for b in BENCHES:
    for pool in POOLS[b]:
        key = f"{b}:{pool}"; P = {}
        g = lambda arm: final_dir(v3g_run(b, arm), arm, pool)
        v = lambda arm: final_dir(f"{b}_v3", arm, pool)
        r5 = lambda arm: final_dir(f"{b}_v3r5", arm, pool)
        P["v3g/dynamic - v3g/S_1"] = compare(g("S_1"), g("dynamic"))
        P["v3g/static - v3g/S_1"] = compare(g("S_1"), g("static"))
        P["v3g/success - v3g/S_1"] = compare(g("S_1"), g("success"))
        P["v3g/dynamic - v3g/static"] = compare(g("static"), g("dynamic"))
        P["v3g/success - v3g/dynamic"] = compare(g("dynamic"), g("success"))
        P["v3/dynamic_sft - v3/S_1"] = compare(v("S_1"), v("dynamic_sft"))
        P["v3/static_sft - v3/S_1"] = compare(v("S_1"), v("static_sft"))
        P["v3/dynamic_sft - v3/static_sft"] = compare(v("static_sft"), v("dynamic_sft"))
        P["v3g/dynamic - v3/dynamic_sft"] = compare(v("dynamic_sft"), g("dynamic"))
        P["v3g/static - v3/static_sft"] = compare(v("static_sft"), g("static"))
        P["v3g/S_1 - v3/S_1"] = compare(v("S_1"), g("S_1"))
        if b != "gpqa":
            P["v3r5/dynamic_sft - v3/S_1"] = compare(v("S_1"), r5("dynamic_sft"))
            P["v3r5/static_sft - v3/S_1"] = compare(v("S_1"), r5("static_sft"))
            P["v3r5/dynamic_sft - v3/dynamic_sft"] = compare(v("dynamic_sft"), r5("dynamic_sft"))
            P["v3r5/static_sft - v3/static_sft"] = compare(v("static_sft"), r5("static_sft"))
            P["v3r5/dynamic_sft - v3r5/static_sft"] = compare(r5("static_sft"), r5("dynamic_sft"))
        pairs[key] = P
        print(f"[compare] {key}: {sum(x is not None for x in P.values())} pairs", flush=True)
data["pairs"] = pairs

# ---------------- 2. GRPO decisions (v3+GRPO runs) ----------------
grpo = {}
for b in BENCHES:
    runs = [f"{b}_v3g"] if b != "mmlu_pro" else ["mmlu_pro_v3g_dyn", "mmlu_pro_v3g_sta", "mmlu_pro_v3g_suc"]
    rows = []
    for run in runs:
        for f in sorted((RUNS / run).glob("r*/**/grpo_dev/decision.json")):
            stage = str(f.relative_to(RUNS / run).parent); d = json.load(open(f)); a = d["accept"]
            rnd = int(stage.split("/")[0][1:]); arm = stage.split("/")[1] if stage.count("/") == 2 else "shared(r1)"
            rows.append({"run": run, "stage": stage, "round": rnd, "arm": arm, "accepted": bool(a["accepted"]), "decision": a["decision"],
                         "reasons": a.get("reasons", []), "d_acc_dev": round(a["deltas"]["accuracy"] * 100, 2) if "deltas" in a else None,
                         "d_calls_dev": round(a["deltas"]["calls_per_example"], 3) if "deltas" in a else None,
                         "grpo_steps": a.get("grpo_steps"), "selected_step": a.get("selected_step"), "sk_acc_dev": round(d["metrics"]["accuracy"] * 100, 2)})
    # informative fraction from report.md "FA-GRPO" table
    for run in runs:
        txt = open(RUNS / run / "report.md").read()
        sec = txt.split("## FA-GRPO")[1].split("\n## ")[0] if "## FA-GRPO" in txt else ""
        for line in sec.splitlines():
            if line.startswith("| r"):
                c = [x.strip() for x in line.strip("|").split("|")]
                st = c[0].replace("/grpo", "/grpo_dev")
                for r in rows:
                    if r["run"] == run and r["stage"] == st:
                        r["informative_frac"] = float(c[5]); r["guard_ok"] = c[4] == "True"; r["steps"] = int(c[2]); r["selected_step"] = int(c[3])
    # per-arm counts: shared r1 + the arm's own r2, r3
    counts = {}
    for arm in ("dynamic", "static", "success"):
        own = [r for r in rows if r["arm"] == arm or (r["arm"] == "shared(r1)" and (b != "mmlu_pro" or r["run"] == {"dynamic": "mmlu_pro_v3g_dyn", "static": "mmlu_pro_v3g_sta", "success": "mmlu_pro_v3g_suc"}[arm]))]
        counts[arm] = {"accepted": sum(r["accepted"] for r in own), "total": len(own), "rounds_accepted": sorted(r["round"] for r in own if r["accepted"])}
    uniq = {(r["run"], r["stage"]): r for r in rows}
    grpo[b] = {"decisions": rows, "per_arm": counts, "accepted_total": sum(r["accepted"] for r in uniq.values()), "total": len(uniq)}
data["grpo"] = grpo

# ---------------- 3. dev trajectories (rounds 1-5, v3 chain; rounds 1-3 v3+GRPO) ----------------
def md_dev_rows(path):
    txt = open(path).read(); sec = txt.split("## Dev evaluations")[1].split("\n## ")[0]; rows = {}
    for line in sec.splitlines():
        if line.startswith("| r"):
            c = [x.strip() for x in line.strip("|").split("|")]
            rows[c[0]] = {"acc": float(c[2]) * 100, "draft_acc": float(c[3]) * 100, "calls": float(c[5]), "correction": float(c[8]), "corruption": float(c[9])}
    return rows
traj = {}
for b in BENCHES:
    T = {"v3": {"dynamic": {}, "static": {}}, "v3g": {"dynamic": {}, "static": {}, "success": {}}, "v3g_sft": {"dynamic": {}, "static": {}, "success": {}}}
    rep = json.load(open(RUNS / f"{b}_v3/report.json"))
    for x in rep["dev"]:
        row = {"acc": x["accuracy"] * 100, "draft_acc": x["initial_draft_accuracy"] * 100, "calls": x["calls_per_example"], "correction": x["correction_rate"], "corruption": x["corruption_rate"]}
        if x["stage"] == "r1/S1_dev":
            T["v3"]["dynamic"][1] = row; T["v3"]["static"][1] = row
        else:
            arm = x["stage"].split("/")[1].replace("_sft", ""); T["v3"][arm][x["round"]] = row
    if b != "gpqa":
        for st, row in md_dev_rows(RUNS / f"{b}_v3r5/report.md").items():
            rnd = int(st.split("/")[0][1:]); arm = st.split("/")[1].replace("_sft", ""); T["v3"][arm][rnd] = row
    runs = [f"{b}_v3g"] if b != "mmlu_pro" else ["mmlu_pro_v3g_dyn", "mmlu_pro_v3g_sta", "mmlu_pro_v3g_suc"]
    for run in runs:
        rows = md_dev_rows(RUNS / run / "report.md"); txt = open(RUNS / run / "report.md").read()
        # resolved checkpoint per round from the Arms tables (G_k, or S_k when GRPO was rejected)
        for m in re.finditer(r"\| (dynamic|static|success): round \| G stage.*?\n\|---.*?\n((?:\|.*\n)+)", txt):
            arm = m.group(1)
            for line in m.group(2).splitlines():
                c = [x.strip() for x in line.strip("|").split("|")]
                T["v3g"][arm][int(c[0])] = {"acc": float(c[3]) * 100, "calls": float(c[4]), "grpo_rejected": c[2] == "True", "stage": c[1]}
        for st, row in rows.items():
            if st.endswith("/sft_dev") and st != "r1/S1_dev":
                arm = st.split("/")[1]; T["v3g_sft"][arm][int(st.split("/")[0][1:])] = row
            if st == "r1/S1_dev":
                for arm in ("dynamic", "static", "success"):
                    T["v3g_sft"][arm][1] = row
    traj[b] = T
data["dev_trajectory"] = traj

# ---------------- 4. tokens per question ----------------
tokens = {}
for b in BENCHES:
    for pool in POOLS[b]:
        g = f"{b}_{pool}"; lab = {}
        for d in ("out", "base"):
            p = TOK / d / g / "summary.json"
            if p.is_file():
                lab.update(json.load(open(p))["labels"])
        merged = TOK / "merged" / g
        if merged.is_dir():
            for f in sorted(merged.glob("v3g__*.jsonl")):
                rs = [json.loads(l) for l in open(f)]; n = len(rs)
                lab[f.stem.replace("__", "/")] = {"accuracy": sum(r["correct"] for r in rs) / n, "calls": sum(r["calls"] for r in rs) / n, "total": sum(r["total"] for r in rs) / n,
                                                   "decode": sum(r["decode"] for r in rs) / n, "mgr_in": sum(r["mgr_in"] for r in rs) / n, "mgr_out": sum(r["mgr_out"] for r in rs) / n,
                                                   "adv_in": sum(r["adv_in"] for r in rs) / n, "adv_out": sum(r["adv_out"] for r in rs) / n, "n": n}
        tokens[f"{b}:{pool}"] = {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "tokens_per_advisor_per_q"} for k, v in lab.items()}
# page-only numbers (token accounting of the pod-4 runs, results page section 11)
tokens["mmlu_pro:test"]["v3g/dynamic"] = {"accuracy": 0.686, "calls": 0.36, "total": 2355, "decode": 322, "src": "results page §11 (pod-4 token accounting; per-question file not archived)"}
tokens["aqua:test"]["v3g/dynamic"] = {"accuracy": 0.776, "calls": 0.77, "total": 3127, "decode": 571, "src": "results page §11 (pod-3 token accounting; per-question file lost with the pod)"}
data["tokens"] = tokens
data["tokens_note"] = ("total = manager prefill + manager output + advisor prefill + advisor output per question (chat template with tool schemas re-rendered per turn, Qwen tokenizer); "
                       "decode = manager output + advisor output. 'base/manager' = untrained base model in the manager seat with tools; 'draft_only' = prompt + DRAFT line only (floor). "
                       "v3g/static and v3g/success tokens were not measured on MMLU-Pro and AQuA; 'abl/success' / 'main/success' are the v1 success runs (same always-call policy).")

# ---------------- 5. base system (untrained manager + tools) ----------------
base_sys = {}
for b in BENCHES:
    for pool in POOLS[b]:
        p = RUNS / "base_manager" / b / pool / "mcq_rsi_eval.json"
        if p.is_file():
            m = json.load(open(p))["metrics"]
            base_sys[f"{b}:{pool}"] = {"acc": round(m["accuracy"] * 100, 2), "calls": round(m["calls_per_example"], 3), "valid": round(m["valid_answer_rate"], 3),
                                      "acc_given_valid": round(m["accuracy"] / m["valid_answer_rate"] * 100, 2), "draft_acc": round(m["initial_draft_accuracy"] * 100, 2), "n": m["n"]}
data["base_system"] = base_sys

# ---------------- 6. single-agent baselines ----------------
single = {}
for v in sorted(os.listdir(SA)):
    if v == "adapters" or not (SA / v).is_dir():
        continue
    for b in BENCHES:
        for pool in POOLS[b]:
            p = SA / v / b / pool / "summary.json"
            if p.is_file():
                s = json.load(open(p))
                single.setdefault(f"{b}:{pool}", {})[v] = {"acc": round(s["accuracy"] * 100, 2), "valid": round(s["valid_answer_rate"], 3), "forced": round(s.get("forced_rate", 0), 3),
                                                            "hit_limit": round(s["hit_limit_rate"], 3), "acc_unforced": round(s.get("accuracy_unforced", s["accuracy"]) * 100, 2),
                                                            "valid_unforced": round(s.get("valid_unforced", s["valid_answer_rate"]), 3), "in_tok": round(s["in_tokens"]), "out_tok": round(s["out_tokens"]),
                                                            "total_tok": round(s["total_tokens"]), "max_new_tokens": s["max_new_tokens"], "mode": s["mode"], "n": s["n"]}
# GPQA base, reasoning allowed (2048 tokens, forced): the pod stopped before its backup; only the summary line in the pod log
# survived (2026-10-11 ~22:05 UTC). Not used by any figure or table since the user dropped the reasoning baselines.
single.setdefault("gpqa:test", {}).setdefault("base_plain_2048f", {"acc": 61.0, "valid": 1.0, "forced": 0.43, "hit_limit": 0.48, "acc_unforced": 45.0, "valid_unforced": 0.57,
    "in_tok": 264, "out_tok": 1530, "total_tok": 1794, "max_new_tokens": 2048, "mode": "plain", "n": 100, "src": "summary line in the pod log only (records lost when the pod stopped)"})
data["single_agent"] = single
data["single_agent_variants"] = {"base_nocot_f": "base model, letter only (16-token budget, forced choice)", "s1_nocot_f": "S_1, letter only", "dyn3_nocot_f": "RSI dynamic r3 (v3, no GRPO), letter only",
                                 "s1_manager_notools": "S_1 with its manager prompt, tools removed (DRAFT then ANSWER, 10 tokens)", "dyn3_manager_notools": "RSI dynamic r3 with the manager prompt, tools removed",
                                 "base_plain": "base, reasoning allowed, 512 tokens, no forcing", "base_plain_512f": "base, reasoning allowed, 512 tokens, forced choice",
                                 "base_plain_2048f": "base, reasoning allowed, 2048 tokens, forced choice", "s1_plain_2048f": "S_1, reasoning allowed, 2048 tokens, forced choice"}

# ---------------- 7. label quality by round + experiment D ----------------
def md_table(path):
    rows = []
    for line in open(path):
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| labels"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            rows.append({"labels": c[0], "sft_rows": None if c[1] == "-" else int(c[1]), "acc": float(c[2]) * 100, "draft_acc": float(c[3]) * 100, "call_rate": float(c[4]),
                         "calls": float(c[5]), "call_rate_draft_wrong": float(c[6]), "correction": float(c[7]), "corruption": float(c[8]), "valid": float(c[9])})
    return rows
lq = {b: md_table(SCR / f"results/summaries/{b}_v3lq.md") for b in BENCHES}
gq = {b: md_table(SCR / f"results/summaries/{b}_v3gq.md") for b in ("mmlu_pro", "gpqa", "aqua")}
data["label_quality"] = lq; data["grpo_data_quality"] = gq
data["label_quality_note"] = ("Each row: SFT from S_1 on that round's labels only (v3 chain labels: r1 = static's data, r_k = re-collected by S_{k-1}), evaluated on dev "
                              "(MedQA/MMLU-Pro 200, GPQA 100, AQuA 254). Experiment D: labels collected on the same questions by G_k (after GRPO) vs S_k (before), both trained from S_1.")

# ---------------- 8. oracle by round (dev), results page §12.2 ----------------
data["oracle_dev"] = {"src": "results page §12.2; per-question best of {commit, one advisor call} from the forced dev evaluations of each run (final/*/dev_forced_*)",
                      "medqa": {"n": 200, "draft_only": 70.5, "policy": {"S_1": 79.0, "r3": 79.0, "r5": 76.0}, "always_verifier": 86.5, "oracle": {"S_1": 92.0, "r3": 92.0, "r5": 92.5}},
                      "mmlu_pro": {"n": 200, "draft_only": 55.5, "policy": {"S_1": 62.5, "r3": 62.5, "r5": 61.5}, "always_verifier": 67.0, "oracle": {"S_1": 79.0, "r3": 77.5, "r5": 76.5}},
                      "gpqa": {"n": 100, "draft_only": 41.0, "policy": {"S_1": 52.0, "r3": 53.0}, "always_verifier": 52.0, "oracle": {"S_1": 61.0, "r3": 63.0}},
                      "aqua": {"n": 254, "draft_only": 39.4, "policy": {"S_1": 73.2, "r3": 77.6, "r5": 72.4}, "always_verifier": 77.2, "oracle": {"S_1": 81.9, "r3": 82.3, "r5": 81.9}}}

# ---------------- 9. self-check (S_1 on dev) ----------------
sc = {}
for b in BENCHES:
    txt = open(SCR / f"results/selfcheck/{b}.md").read()
    auroc = {}; router = {}
    for line in txt.splitlines():
        m = re.match(r"\| (1-p_max|-margin|entropy|1-agreement) \| ([0-9.]+) \| ([-0-9.]+) \| ([-0-9.]+) \|$", line)
        if m: auroc[m.group(1)] = float(m.group(2))
        m2 = re.match(r"\| (1-p_max|-margin|entropy|1-agreement) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+), ([0-9.]+) \|$", line)
        if m2: router[m2.group(1)] = {"acc_at_policy_budget": float(m2.group(2)) * 100, "acc_at_alt_budget": float(m2.group(3)) * 100, "best_calls": float(m2.group(4)), "best_acc": float(m2.group(5)) * 100}
    pol = re.search(r"policy on dev: acc ([0-9.]+) at calls ([0-9.]+); always-Verifier ([0-9.]+)", txt)
    alt = re.search(r"acc @ calls ([0-9.]+) \| acc @ calls ([0-9.]+)", txt)
    sc[b] = {"auroc": auroc, "router": router, "policy_acc": float(pol.group(1)) * 100, "policy_calls": float(pol.group(2)), "always_verifier": float(pol.group(3)) * 100,
             "alt_budget": float(alt.group(2)), "n": N_DEV[b]}
data["self_check"] = sc

# ---------------- 10. matched-budget replay (v3 no-GRPO, dev) + page notes ----------------
rep = {}
for b in BENCHES:
    rows = []
    with open(RUNS / f"{b}_v3/analysis/matched_budget_replay.csv") as f:
        for r in csv.DictReader(f):
            lo, hi = json.loads(r["random 95% interval"])
            rows.append({"label": r["label"], "n": int(r["n"]), "roles": json.loads(r["roles"]), "replay_acc": round(float(r["policy replay acc"]) * 100, 1), "random_lo": lo * 100, "random_hi": hi * 100,
                         "p": float(r["p"]), "above": r["above interval"] == "True"})
    rep[b] = rows
data["replay_v3"] = rep
data["replay_notes"] = {"mmlu_pro v3g/dynamic (dev)": {"replay_acc": 64.5, "random_lo": 54.5, "random_hi": 61.5, "p": "<0.001", "src": "results page §10"},
                        "medqa v3r5/dynamic_sft": {"replay_acc": 76.0, "random_lo": 73.5, "random_hi": 77.5, "p": 0.37, "src": "results page §10"},
                        "medqa v3r5/static_sft": {"replay_acc": 79.0, "random_lo": 72.0, "random_hi": 78.5, "p": 0.015, "src": "results page §10"},
                        "mmlu_pro v3r5/dynamic_sft": {"replay_acc": 60.5, "random_lo": 53.0, "random_hi": 59.0, "p": 0.003, "src": "results page §10"},
                        "aqua v3r5/dynamic_sft": {"p": 0.15, "above": False, "src": "results page §10"}, "aqua v3r5/static_sft": {"p": 0.002, "above": True, "src": "results page §10"}}

# ---------------- 11. setup ----------------
cfg = json.load(open(REPO / "configs/mcq_rsi_medqa_v3.json"))
data["setup"] = {"base_model": "Qwen3.5-9B (non-thinking; enable_thinking=False everywhere)", "advisors": "base model + role prompts (Extractor, Reasoner, Verifier), served with vLLM, outputs cached",
                 "manager": "LoRA on Qwen3.5-9B; S_1 = the paper's one-round SFT manager", "rounds": 3, "extra_rounds": "4-5 for the no-GRPO v3 chain (MedQA, MMLU-Pro, AQuA)",
                 "collect": {"questions_per_round": 400, "max_depth": 2, "gpqa": "reuses its 200 round-1 roots"}, "rho_commit_rescue": RHO,
                 "sft": cfg["sft"], "grpo": cfg["grpo"], "grpo_questions_per_round": 256, "grpo_gpqa_pool": 146, "gates": cfg["gates"],
                 "grpo_acceptance": "dev acc >= S_k - 1 pt; calls <= S_k + 0.15; call gap >= 0.5 x S_k gap; >= 15% informative revision states (J in (0.02, 0.98)); else G_k := S_k",
                 "grpo_guard": "every 8 steps on a 64-question probe: call-rate delta <= 0.10 and KL_dec <= 0.05, otherwise roll back to the last good step",
                 "arms": {"dynamic": "re-collect with G_{k-1} each round, marginal-value labels, SFT from G_{k-1}, then FA-GRPO", "static": "reuse S_1's labels every round, SFT from G_{k-1}, then FA-GRPO",
                          "success": "like dynamic but any trajectory ending in a correct answer is a demonstration (shortest path not required)"},
                 "pools": {"dev": N_DEV, "locked_test": N_TEST}, "versions": {"v1": "paper SFT target (draft letter supervised) + GRPO", "v3": "draft letter never supervised (route-only loss), no GRPO", "v3+GRPO (main)": "v3 loss + FA-GRPO each round"}}

json.dump(data, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT, OUT.stat().st_size, "bytes")

# ---------------- consistency checks against the results page ----------------
PAGE = {"medqa:test": {"v3g/S_1": (78.8, .37), "v3g/dynamic": (78.6, .17), "v3g/static": (79.8, .28), "v3g/success": (80.2, .99), "v3/S_1": (78.4, .36), "v3/dynamic_sft": (78.6, .15), "v3/static_sft": (78.6, .27), "v3r5/dynamic_sft": (77.4, .17), "v3r5/static_sft": (81.0, .37)},
        "mmlu_pro:test": {"v3g/S_1": (63.6, .35), "v3g/dynamic": (68.6, .36), "v3g/static": (61.4, .26), "v3g/success": (71.8, 1.0), "v3/S_1": (64.0, .35), "v3/dynamic_sft": (64.4, .27), "v3/static_sft": (58.0, .26), "v3r5/dynamic_sft": (61.4, .30), "v3r5/static_sft": (58.0, .25)},
        "mmlu_pro:test_paper": {"v3g/S_1": (59.6, .35), "v3g/dynamic": (64.6, .39), "v3g/static": (58.0, .28), "v3g/success": (69.0, 1.0), "v3/S_1": (59.6, .35), "v3/dynamic_sft": (59.8, .27), "v3/static_sft": (59.0, .29), "v3r5/dynamic_sft": (58.6, .29), "v3r5/static_sft": (53.8, .26)},
        "gpqa:test": {"v3g/S_1": (54.0, .52), "v3g/dynamic": (55.0, .60), "v3g/static": (50.0, .47), "v3g/success": (58.0, 1.0), "v3/S_1": (54.0, .52), "v3/dynamic_sft": (54.0, .58), "v3/static_sft": (55.0, .49)},
        "aqua:test": {"v3g/S_1": (77.2, .78), "v3g/dynamic": (77.6, .77), "v3g/static": (73.6, .72), "v3g/success": (82.7, 1.0), "v3/S_1": (77.6, .78), "v3/dynamic_sft": (78.0, .79), "v3/static_sft": (74.8, .73), "v3r5/dynamic_sft": (71.7, .72), "v3r5/static_sft": (77.2, .76)}}
bad = 0
for key, exp in PAGE.items():
    for lab, (acc, calls) in exp.items():
        m = locked[key].get(lab)
        if m is None:
            print(f"MISSING {key} {lab}"); bad += 1; continue
        if abs(m["acc"] - acc) > 0.06 or abs(m["calls"] - calls) > 0.006:
            print(f"MISMATCH {key} {lab}: file {m['acc']}/{m['calls']} vs page {acc}/{calls}"); bad += 1
PAGE_P = {"mmlu_pro:test": {"v3g/dynamic - v3g/S_1": (5.0, .003), "v3g/dynamic - v3/dynamic_sft": (4.2, .003), "v3g/static - v3/static_sft": (3.4, .02), "v3g/success - v3g/S_1": (8.2, .001)},
          "medqa:test": {"v3g/dynamic - v3g/S_1": (-0.2, .94), "v3g/static - v3/static_sft": (1.2, .13), "v3r5/dynamic_sft - v3r5/static_sft": (-3.6, .008), "v3r5/static_sft - v3/S_1": (2.6, .02)},
          "aqua:test": {"v3g/success - v3g/S_1": (5.5, .016), "v3r5/dynamic_sft - v3/S_1": (-5.9, .009), "v3g/dynamic - v3/dynamic_sft": (-0.4, .93)},
          "gpqa:test": {"v3g/dynamic - v3g/S_1": (1.0, .84), "v3g/static - v3/static_sft": (-5.0, .12)}}
for key, exp in PAGE_P.items():
    for lab, (d, p) in exp.items():
        c = pairs[key].get(lab)
        if c is None:
            print(f"MISSING pair {key} {lab}"); bad += 1; continue
        flag = "" if abs(c["d_acc"] - d) <= 0.06 and abs(c["p"] - p) <= 0.02 else "  <-- differs from page"
        print(f"[pair] {key} {lab}: Δ{c['d_acc']:+.1f} p={c['p']:.3f} (page Δ{d:+.1f} p={p})" + flag)
        if flag: bad += 1
for b in BENCHES:
    print(f"[grpo] {b}: accepted {grpo[b]['accepted_total']}/{grpo[b]['total']}  per arm " + ", ".join(f"{a} {c['accepted']}/{c['total']} r{c['rounds_accepted']}" for a, c in grpo[b]["per_arm"].items()))
print("checks:", "OK" if bad == 0 else f"{bad} problems")
