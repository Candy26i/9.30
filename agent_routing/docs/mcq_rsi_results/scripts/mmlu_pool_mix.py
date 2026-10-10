# Why does the MMLU-Pro collect-pool oracle differ by round? Category mix of each pool and the per-category oracle,
# from the pods' normalized data cache and the collection records of the v3 / v3r5 runs.
from huggingface_hub import HfApi, hf_hub_download
import json, glob, collections
cat = {}
for p in glob.glob("/workspace/9.30/agent_routing/outputs/data/mmlu_pro*normalized*.jsonl"):
    for l in open(p):
        r = json.loads(l); cat[str(r.get("example_id"))] = r.get("category") or r.get("subject") or r.get("task_subtype") or "?"
print("normalized rows with a category:", len(cat), "sample:", list(cat.items())[:2])
m = json.load(open("/workspace/9.30/agent_routing/data/mcq_rsi/mmlu_pro_splits_r5.json"))
api = HfApi()
recs = {}
for run, rounds in (("mmlu_pro_v3", {2: "r2/collect", 3: "r3/dynamic_sft/collect"}), ("mmlu_pro_v3r5", {4: "r4/dynamic_sft/collect", 5: "r5/dynamic_sft/collect"})):
    files = api.list_repo_files(f"MaliDDD/margent-mcq-rsi-{run}")
    for k, stage in rounds.items():
        path = f"runs/{run}/{stage}/counterfactual_records.jsonl"
        if path not in files:
            print("missing", path); continue
        p = hf_hub_download(f"MaliDDD/margent-mcq-rsi-{run}", path, local_dir="/workspace/tmp/oracle_dl")
        recs[k] = [json.loads(l) for l in open(p) if l.strip()]
k0 = min(recs); print("record keys:", sorted(recs[k0][0])[:40])

def any_correct(x):
    if isinstance(x, dict):
        if x.get("correct") is True or str(x.get("correct")) == "True":
            return True
        return any(any_correct(v) for v in x.values())
    if isinstance(x, list):
        return any(any_correct(v) for v in x)
    return False
print("branches sample:", json.dumps(recs[k0][0]["branches"])[:300])
mix, orc, direct = {}, {}, {}
for k, rs in sorted(recs.items()):
    c = collections.Counter(cat.get(str(r["example_id"]), "?") for r in rs)
    o = collections.defaultdict(list); d = collections.defaultdict(list)
    for r in rs:
        cc = cat.get(str(r["example_id"]), "?")
        o[cc].append(bool(r.get("direct_correct")) or any_correct(r["branches"])); d[cc].append(bool(r.get("direct_correct")))
    mix[k], orc[k], direct[k] = c, {cc: sum(v) / len(v) for cc, v in o.items()}, {cc: sum(v) / len(v) for cc, v in d.items()}
    tot = sum(sum(v) for v in o.values()) / len(rs)
    print(f"r{k}: oracle={tot*100:.1f} direct={sum(sum(v) for v in d.values())/len(rs)*100:.1f} mix={dict(c.most_common(14))}")
# reweight every round's per-category oracle to the round-2 category mix
base = mix[min(recs)]
for k in sorted(recs):
    w = sum(base[cc] * orc[k].get(cc, orc[k].get("?", 0)) for cc in base) / sum(base.values())
    print(f"r{k}: oracle reweighted to r{min(recs)} mix = {w*100:.1f}")
cats = sorted(set().union(*[set(v) for v in orc.values()]))
print("per-category oracle (r2/r3/r4/r5):")
for cc in cats:
    print(f"  {cc:18}", " ".join(f"{orc[k].get(cc, float('nan'))*100:5.1f}(n={mix[k].get(cc,0):2d})" for k in sorted(recs)))
