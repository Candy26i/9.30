# Local poller: short SSH calls every 120 s (transient SSH failures are retried); exits on any change of controller
# state / liveness / final-test exit, or after max_s. Robust to the flaky long-lived SSH connection.
import json, os, subprocess, sys, time
max_s = float(sys.argv[1]) if len(sys.argv) > 1 else 3000
HOST, PORT = os.environ.get("POD_HOST", "root@<pod-ip>"), os.environ.get("POD_PORT", "22")  # the ssh key comes from ~/.ssh/config
CMD = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", HOST, "-p", PORT, "python3 /workspace/tmp/state_now.py"]

def state():
    for _ in range(3):
        try:
            r = subprocess.run(CMD, capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            pass
        time.sleep(15)
    return None

start = time.time()
first = None
while first is None and time.time() - start < max_s:
    first = state()
fails = 0
while time.time() - start < max_s:
    time.sleep(120)
    now = state()
    if now is None:
        fails += 1
        continue
    key = lambda v: (v[0], v[2], v[3])
    changed = {k: (first.get(k), v) for k, v in now.items() if k not in first or key(first[k]) != key(v)}
    if changed:
        print("CHANGED", json.dumps(changed)); print("NOW", json.dumps(now)); sys.exit(0)
print("NO CHANGE", json.dumps(now if 'now' in dir() and now else first), "ssh_failures", fails)
