# Print the current state of every run directory as JSON (controller, stage, controller alive, final_test_exit).
import glob, json, os
R = "/workspace/mcq_rsi/runs"
out = {}
for p in sorted(glob.glob(f"{R}/*/status.json")):
    name = os.path.basename(os.path.dirname(p))
    if "." in name:
        continue
    try:
        s = json.load(open(p))
    except Exception:
        continue
    try:
        os.kill(int(s.get("controller_pid")), 0)
        alive = True
    except Exception:
        alive = False
    exitf = f"{R}/{name}/final_test_exit.txt"
    out[name] = [s.get("controller"), s.get("current_stage"), alive, open(exitf).read().strip() if os.path.exists(exitf) else None]
print(json.dumps(out, sort_keys=True))
