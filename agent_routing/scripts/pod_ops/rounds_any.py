# Dev metrics of every sft_dev stage of the given runs (r*/<arm>/sft_dev/decision.json), one line per stage.
import glob, json, os, sys
R = "/workspace/mcq_rsi/runs"
for run in sys.argv[1:]:
    for f in sorted(glob.glob(f"{R}/{run}/r*/*/sft_dev/decision.json")):
        rnd, arm = f.split("/")[-4], f.split("/")[-3]
        d = json.load(open(f)); m = d.get("metrics") or {}
        if not m:
            continue
        print(f"{run:14} {rnd} {arm:12} acc {m['accuracy']:.3f} draft {m['initial_draft_accuracy']:.3f} call {m['call_rate']:.3f} "
              f"cr|wrong {m.get('call_rate_given_draft_wrong', float('nan')):.3f} corr {m['correction_rate']:.3f} corrupt {m['corruption_rate']:.3f} "
              f"valid {m['valid_answer_rate']:.3f} adv {m['per_advisor_calls']}" + ("  [ack]" if d.get("acknowledged_gate") else ""))
    f = f"{R}/{run}/r1/S1_dev/decision.json"
    if os.path.isfile(f):
        m = json.load(open(f)).get("metrics") or {}
        if m:
            print(f"{run:14} r1 S1_dev       acc {m['accuracy']:.3f} draft {m['initial_draft_accuracy']:.3f} call {m['call_rate']:.3f} adv {m['per_advisor_calls']}")
