# Exit (and report) as soon as any watched run or final-test changes to a terminal state, or after max_s.
import json, os, sys, time
R = "/workspace/mcq_rsi/runs"
max_s = float(sys.argv[1]) if len(sys.argv) > 1 else 3300
def state():
    out = {}
    for b in ("medqa", "mmlu_pro", "gpqa", "aqua"):
        try:
            s = json.load(open(f"{R}/{b}_main/status.json"))
        except Exception:
            continue
        alive = True
        try:
            os.kill(int(s.get("controller_pid")), 0)
        except Exception:
            alive = False
        exitf = f"{R}/{b}_main/final_test_exit.txt"
        out[b] = (s.get("controller"), s.get("current_stage"), alive,
                  open(exitf).read().strip() if os.path.exists(exitf) else None)
    return out
start, first = time.time(), state()
while time.time() - start < max_s:
    time.sleep(30)
    now = state()
    changed = {b: (first.get(b), now[b]) for b in now if first.get(b) is None or
               first[b][0] != now[b][0] or first[b][2] != now[b][2] or first[b][3] != now[b][3]}
    bad = {b: v for b, v in now.items() if v[0] in ("failed", "interrupted", "deadline") or (v[0] in ("running", "final-test") and not v[2])}
    if changed or bad:
        print("CHANGED", json.dumps(changed), "BAD", json.dumps(bad)); print("NOW", json.dumps(now)); sys.exit(0)
print("NO CHANGE", json.dumps(state()))
