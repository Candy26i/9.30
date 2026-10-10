# Per-unit label composition and dev metrics of a label-quality run (summary.json of scripts/mcq_label_quality.py).
import json, sys
for run in sys.argv[1:]:
    s = json.load(open(f"/workspace/mcq_rsi/runs/{run}/summary.json"))
    print("==", run)
    for tag, e in s["units"].items():
        ls = e.get("label_stats") or {}; m = e["metrics"]
        print("%-4s rows=%-4s types=%s rescued=%s commit=%s rescue=%s direct=%s oracle=%s | acc=%.3f calls=%.3f adv=%s %ss" % (
            tag, ls.get("rows", "-"), ls.get("decision_types"), ls.get("n_rescued"), ls.get("n_selected_commit_decisions"),
            ls.get("n_selected_rescue_decisions"), ls.get("direct_accuracy"), ls.get("oracle_accuracy"), m["accuracy"],
            m["calls_per_example"], e.get("per_advisor_calls"), e["seconds"]))
