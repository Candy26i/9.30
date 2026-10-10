# Dev-set oracle from the forced-delegation evals of the v3 runs (HF backups): per question, the best of "commit the draft"
# and "call advisor X" (one call) or "call all three" (E+R+V), next to the free policy's accuracy on the same dev set.
from huggingface_hub import HfApi, hf_hub_download
import json, re
api = HfApi()
tb = lambda v: v is True or str(v) == "True"
def rows(repo, path):
    p = hf_hub_download(repo, path, local_dir="/workspace/tmp/oracle_dl")
    return {int(r["example_id"]): r for r in (json.loads(l) for l in open(p) if l.strip())}
def draft(r):
    m = re.search(r"DRAFT_ANSWER_([A-Z])", r.get("final_text") or "")
    return m.group(1) if m else None
import sys
for b in ("medqa", "mmlu_pro", "aqua"):
    run = f"{b}_v3r5"; repo = f"MaliDDD/margent-mcq-rsi-{run}"
    files = api.list_repo_files(repo)
    for lab, free_path in (("dynamic_sft", f"runs/{run}/r5/dynamic_sft/sft_dev/manager_tool_eval.jsonl"),
                           ("static_sft", f"runs/{run}/r5/static_sft/sft_dev/manager_tool_eval.jsonl")):
        free = rows(repo, free_path)
        forced = {}
        for kind in ("extractor", "reasoner", "verifier", "extractor+reasoner+verifier"):
            cand = [f for f in files if f.startswith(f"runs/{run}/final/{lab}/dev_forced_{kind}/") and f.endswith(".jsonl")]
            if cand:
                forced[kind] = rows(repo, cand[0])
        ids = sorted(set(free) & set.intersection(*[set(v) for v in forced.values()]))
        n = len(ids)
        draft_ok = {i: tb(free[i].get("initial_draft_correct")) for i in ids}
        singles = [k for k in forced if k != "extractor+reasoner+verifier"]
        acc = {k: sum(tb(v[i]["correct"]) for i in ids) / n for k, v in forced.items()}
        policy = sum(tb(free[i]["correct"]) for i in ids) / n
        calls = sum(float(free[i].get("tool_calls") or 0) for i in ids) / n
        oracle1 = sum(draft_ok[i] or any(tb(forced[k][i]["correct"]) for k in singles) for i in ids) / n
        oracle_any = sum(draft_ok[i] or any(tb(forced[k][i]["correct"]) for k in forced) for i in ids) / n
        best = max((acc[k], k) for k in singles)
        print(f"{b:9} r5 {lab:12} n={n} draft={sum(draft_ok.values())/n*100:.1f} policy={policy*100:.1f} (calls {calls:.2f}) "
              f"forced: " + " ".join(f"{k[:3]}={acc[k]*100:.1f}" for k in forced) + f" | best_single={best[0]*100:.1f}({best[1][:3]}) "
              f"oracle_1call={oracle1*100:.1f} oracle_any={oracle_any*100:.1f}")
