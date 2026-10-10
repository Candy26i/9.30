from huggingface_hub import HfApi, hf_hub_download
import json
api = HfApi(); repo = "MaliDDD/margent-mcq-rsi"
files = api.list_repo_files(repo)
for run in ("mmlu_pro_main", "aqua_main", "medqa_main", "gpqa_main"):
    p = hf_hub_download(repo, f"runs/{run}/report.json", local_dir="/workspace/tmp/v1reports")
    r = json.load(open(p))
    print("==", run)
    decs = r.get("decisions")
    items = decs.items() if isinstance(decs, dict) else [(d.get("stage") or d.get("name"), d) for d in decs]
    for name, d in items:
        if isinstance(d, dict) and d.get("role") == "grpo":
            a = d.get("accept") or {}
            print("   %-28s accepted=%s steps=%s sel=%s reasons=%s" % (name, a.get("accepted"), a.get("grpo_steps"), a.get("selected_step"), [x[:70] for x in a.get("reasons", [])]))
    if not any(isinstance(d, dict) and d.get("role") == "grpo" for _, d in items):
        print("   (no grpo decisions found; sample:", json.dumps(items[0][1] if items else None)[:300], ")")
print("-- files in mmlu_pro_main r2/dynamic/sft/model and grpo/final")
for d in ("runs/mmlu_pro_main/r2/dynamic/sft/model/", "runs/mmlu_pro_main/r2/dynamic/grpo/final/", "runs/mmlu_pro_main/r2/dynamic/grpo/"):
    print(d, sorted(f.replace(d, "") for f in files if f.startswith(d) and f.count("/") == d.count("/")))
