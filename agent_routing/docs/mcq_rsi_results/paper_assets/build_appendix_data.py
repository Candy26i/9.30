#!/usr/bin/env python3
"""appendix.json: the extra numbers for the appendix tables, each traced to its source.
  advisor_ab   advisor A/B on dev (fixed base solver drafts, reads one advisor output, answers again; n = 200/200/100/254):
               base model with role prompt vs the paper's advisor LoRA (correctly loaded) vs retrained v2 LoRA
               (HF MaliDDD/margent-mcq-rsi logs/advisor_ab/<bench>_summary_s4096_arms-v2.json), plus pooled sign tests
  lora_check   vLLM 0.26 drops the paper LoRA weights: outputs of the "LoRA" server vs the base model on 20 prompts
               (logs/d14_evidence/summary.json)
  draft        draft letters on the locked tests: v1 (paper SFT target, drafts supervised) vs v3 (route-only), from the
               per-question records (v1: HF runs/{medqa,aqua}_main, aqua_r5; v3: locked-test archive). The v1 MedQA
               round-5 records were not backed up; its row uses the values computed on the pod (results page section 7.1).
  mix          which advisors each policy calls on the locked tests (calls per 100 questions), from the records
  tokens       per-question token components (data.json) and advisor-side tokens per call by role, from the base-system runs
Re-run needs the scratchpad copies named below; in the repo, appendix.json is the frozen output."""
import json, math, re
from collections import Counter
from pathlib import Path
SCR = Path("/private/tmp/claude-501/-Users-madili-Desktop-7-98/17a9c165-9b16-40c6-bd20-a405349d1016/scratchpad")
ROOT = Path(__file__).resolve().parent
D = json.load(open(ROOT / "data.json"))
B = ["medqa", "mmlu_pro", "gpqa", "aqua"]
A = {"meta": {"note": "accuracy in %, calls per question; see build_appendix_data.py docstring for sources"}}

def sign_p(a, b):
    n = a + b
    if n == 0: return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

# ---------------- advisor A/B
ab = {}; pooled = {}
for b in B:
    j = json.load(open(SCR / f"hf_ab/{b}_summary_s4096_arms-v2.json"))
    ab[b] = {"n": j["n"], "solver_draft_acc": round(j["draft_accuracy"] * 100, 2), "kinds": {}}
    for kind, kd in j["kinds"].items():
        row = {arm: {"acc": kd[arm]["accuracy"] * 100, "corrections": kd[arm]["corrections"], "corruptions": kd[arm]["corruptions"]} for arm in ("base", "lora", "v2") if arm in kd}
        for arm in ("lora", "v2"):
            vb = kd.get("vs_base", {}).get(arm)
            if vb:
                row[arm].update({"minus_base_pp": vb["minus_base_pp"], "arm_only": vb["arm_only_correct"], "base_only": vb["base_only_correct"], "sign_p": round(vb["sign_test_p"], 4)})
                P = pooled.setdefault(kind, {}).setdefault(arm, {"arm_only": 0, "base_only": 0, "n": 0})
                P["arm_only"] += vb["arm_only_correct"]; P["base_only"] += vb["base_only_correct"]; P["n"] += j["n"]
        ab[b]["kinds"][kind] = row
for kind, arms in pooled.items():
    for arm, P in arms.items():
        P["minus_base_pp"] = round((P["arm_only"] - P["base_only"]) / P["n"] * 100, 2); P["sign_p"] = round(sign_p(P["arm_only"], P["base_only"]), 4)
A["advisor_ab"] = {"per_bench": ab, "pooled": pooled}
d14 = json.load(open(SCR / "hf_ab/d14_summary.json"))
A["lora_check"] = {k: {"original_vs_base_identical": v["original vs base"]["identical"], "original_vs_base_similarity": v["original vs base"]["median_similarity"],
                       "renamed_vs_base_identical": v["renamed vs base"]["identical"], "renamed_vs_base_similarity": v["renamed vs base"]["median_similarity"], "n_prompts": 20} for k, v in d14.items()}

# ---------------- draft letters (locked test)
def recs(path):
    return [json.loads(l) for l in open(path) if l.strip()]
def tb(v): return v is True or str(v).lower() == "true"
def tools(r):
    t = r["tool_names_called"]; t = t if isinstance(t, list) else json.loads(str(t).replace("'", '"'))
    return [x.replace("_tool", "") for x in t]
def summarize(rows):
    n = len(rows); dr = Counter(r["initial_draft"] for r in rows); gt = Counter(r["ground_truth"] for r in rows)
    mix = Counter(); [mix.update(tools(r)) for r in rows]
    return {"n": n, "draft_B_pct": dr.get("B", 0) / n * 100, "truth_B_pct": gt.get("B", 0) / n * 100,
            "draft_acc": sum(tb(r["initial_draft_correct"]) for r in rows) / n * 100, "acc": sum(tb(r["correct"]) for r in rows) / n * 100,
            "calls": sum(int(r["tool_calls"]) for r in rows) / n, "per100": {k: mix.get(k, 0) / n * 100 for k in ("extractor", "reasoner", "verifier")}}
V1 = SCR / "hf_v1/runs"; ARCH = SCR / "locked_archive/runs"; RUNS = SCR / "results/runs"
def find(run, arm, pool="test"):
    for base in (RUNS, ARCH, V1):
        p = base / run / "final" / arm / pool / "manager_tool_eval.jsonl"
        if p.is_file(): return p
    return None
draft = {}
for b in ("medqa", "aqua"):
    rows = {}
    spec = [("S_1", f"{b}_v3", "S_1"), ("v1 dynamic R3", f"{b}_main", "dynamic"), ("v1 static R3", f"{b}_main", "static"),
            ("v1 dynamic R5", f"{b}_r5", "dynamic"), ("v1 static R5", f"{b}_r5", "static"),
            ("v3 dynamic R3", f"{b}_v3", "dynamic_sft"), ("v3 static R3", f"{b}_v3", "static_sft"),
            ("v3 dynamic R5", f"{b}_v3r5", "dynamic_sft"), ("v3 static R5", f"{b}_v3r5", "static_sft")]
    for name, run, arm in spec:
        p = find(run, arm)
        rows[name] = dict(summarize(recs(p)), src=str(p.relative_to(SCR))) if p else None
    draft[b] = rows
# v1 MedQA round 5: records not backed up; values computed on the pod (results page section 7.1 / 8; accuracy from token accounting)
TK = D["tokens"]["medqa:test"]
draft["medqa"]["v1 dynamic R5"] = {"n": 500, "draft_B_pct": round(213 / 500 * 100, 2), "truth_B_pct": round(113 / 500 * 100, 2), "draft_acc": 67.4,
                                   "acc": round(TK["r5/dynamic"]["accuracy"] * 100, 2), "calls": round(TK["r5/dynamic"]["calls"], 3), "per100": None,
                                   "src": "pod computation (results page 7.1: 213 B drafts of 500, draft accuracy 67.4%); accuracy/calls from data.json tokens r5/dynamic"}
draft["medqa"]["v1 static R5"] = {"n": 500, "draft_B_pct": round(135 / 500 * 100, 2), "truth_B_pct": round(113 / 500 * 100, 2), "draft_acc": None,
                                  "acc": round(TK["r5/static"]["accuracy"] * 100, 2), "calls": round(TK["r5/static"]["calls"], 3), "per100": None,
                                  "src": "pod computation (results page 7.1: 135 B drafts of 500); accuracy/calls from data.json tokens r5/static"}
A["draft"] = draft

# ---------------- advisor mix on the locked tests (v3 chain and v3+GRPO)
mix = {}
for b in B:
    m = {}
    for name, run, arm in [("S_1", f"{b}_v3", "S_1"), ("Static R3", f"{b}_v3", "static_sft"), ("Dynamic R3", f"{b}_v3", "dynamic_sft"),
                           ("Static R5", f"{b}_v3r5", "static_sft"), ("Dynamic R5", f"{b}_v3r5", "dynamic_sft"),
                           ("Static R3 + GRPO", f"{b}_v3g" if b != "mmlu_pro" else "mmlu_pro_v3g_sta", "static"),
                           ("Dynamic R3 + GRPO", f"{b}_v3g" if b != "mmlu_pro" else "mmlu_pro_v3g_dyn", "dynamic"),
                           ("Success R3 + GRPO", f"{b}_v3g" if b != "mmlu_pro" else "mmlu_pro_v3g_suc", "success")]:
        p = find(run, arm)
        if p:
            s = summarize(recs(p)); m[name] = {"per100": s["per100"], "calls": s["calls"], "acc": s["acc"], "n": s["n"]}
    mix[b] = m
A["mix"] = mix

# ---------------- tokens: components and advisor-side tokens per call (base system: every role is called often)
tok = {}
for b in B:
    t = D["tokens"][f"{b}:test"]; row = {}
    for name, keys in [("Base system", ["base/manager"]), ("S_1", ["v3g/S_1", "v3/S_1"]), ("RSI dynamic", ["v3g/dynamic"]), ("Success", ["v3g/success", "abl/success", "main/success"])]:
        k = next((k for k in keys if k in t), None)
        if k is None: continue
        x = t[k]; row[name] = {"label": k, **{c: x.get(c) for c in ("mgr_in", "mgr_out", "adv_in", "adv_out", "total", "decode")}, "calls": x["calls"], "n": x.get("n")}
    s = json.load(open(SCR / f"results/runs/base_manager/tokcost/{b}_test/summary.json"))["labels"]["base/manager"]
    m = json.load(open(SCR / f"results/runs/base_manager/{b}/test/mcq_rsi_eval.json"))["metrics"]
    pac = m["per_advisor_calls"]; per_call = {}
    for adv, tpq in s["tokens_per_advisor_per_q"].items():
        c = pac.get(f"{adv}_tool", 0)
        per_call[adv] = tpq * s["n"] / c if c else None
    tok[b] = {"rows": row, "advisor_tokens_per_call": per_call, "per_call_src": "base/manager token accounting: advisor in+out tokens per question attributed to each role x n / that role's call count"}
A["tokens"] = tok
json.dump(A, open(ROOT / "appendix.json", "w"), indent=1)
print("wrote appendix.json")
# quick look
for kind, arms in pooled.items(): print("pooled", kind, {a: (v["minus_base_pp"], v["arm_only"], v["base_only"], v["sign_p"]) for a, v in arms.items()})
for b in ("medqa", "aqua"):
    for k, v in draft[b].items(): print(b, f"{k:15s}", None if v is None else (v["draft_B_pct"], v["truth_B_pct"], v["draft_acc"], v["acc"], v["calls"]))
for b in B: print(b, "per-call", tok[b]["advisor_tokens_per_call"])
