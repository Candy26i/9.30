# Draft-letter distribution and advisor mix of every locked-test eval of the given runs.
import glob, json, os, sys
from collections import Counter
b = lambda v: v is True or str(v).lower() == "true"
for run in sys.argv[1:]:
    for p in sorted(glob.glob(f"/workspace/mcq_rsi/runs/{run}/final/*/test*/manager_tool_eval.jsonl")):
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if not rows:
            continue
        lab, pool = p.split("/")[-3], p.split("/")[-2]
        dr = Counter(r["initial_draft"] for r in rows); gt = Counter(r["ground_truth"] for r in rows)
        tl = Counter()
        for r in rows:
            t = r["tool_names_called"]
            t = t if isinstance(t, list) else json.loads(t.replace("'", '"'))
            tl.update(x.replace("_tool", "")[0].upper() for x in t)
        n = len(rows)
        print(f"{run} {lab}/{pool}: n={n} acc={sum(b(r['correct']) for r in rows)/n:.3f} draft={sum(b(r['initial_draft_correct']) for r in rows)/n:.3f} "
              f"calls={sum(int(r['tool_calls']) for r in rows)/n:.3f} drafts={dict(sorted(dr.items()))} truth={dict(sorted(gt.items()))} E/R/V={tl.get('E',0)}/{tl.get('R',0)}/{tl.get('V',0)}")
