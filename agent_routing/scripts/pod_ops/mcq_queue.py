#!/usr/bin/env python3
"""Overnight MCQ RSI scheduler (operator decisions of 2026-10-03).

Keeps up to MAX_CONCURRENT main runs going on the pod. MedQA (already running, 3 arms) counts as one;
the queue starts mmlu_pro, gpqa, aqua (arms dynamic,static) as slots free. Per run:
  - completed                         -> slot freed
  - failed at the round-1 parity gate -> auto-acknowledged when every failing gated check is an accuracy
                                         check within 4 pt (calls etc. must pass), then resumed; else stopped
  - any other failure / dead controller -> resumed once (transient advisor/OOM errors), then stopped
Queue items "final:<bench>" run that benchmark's locked final-test once (no retry; a failure waits for the
operator) and hold a slot while they run (GPU 0 memory: vLLM + two inference managers at most).
Never edits the repository (the run signature binds git HEAD). Log: /workspace/mcq_rsi/logs/queue.log
"""
import json
import os
import subprocess
import time
from pathlib import Path

WORK = Path("/workspace/mcq_rsi")
REPO = Path("/workspace/9.30/agent_routing")
PY = "/workspace/mcq-venv/bin/python"
URL = "http://127.0.0.1:18002"
MAX_CONCURRENT = 2
PARITY_MAX_ACC_GAP = 0.04
QUEUE = ["mmlu_pro", "gpqa", "aqua"]
RUNS = {"medqa": {"arms": None}}  # medqa: config arms (3), started earlier by the wrapper
ARMS = "dynamic,static"
ENV = {**os.environ, "HF_HOME": "/workspace/hf-cache", "HF_HUB_DISABLE_XET": "1", "TMPDIR": "/workspace/tmp",
       "MCQ_ADVISOR_PORT": "18002", "PYTHONUNBUFFERED": "1"}
ENV.pop("MARGENT_WANDB_MODE", None)
STATE = WORK / "logs" / "queue_state.json"


def log(msg):
    line = f"[mcq-queue {time.strftime('%F %T')}] {msg}"
    print(line, flush=True)


def run_dir(bench):
    return WORK / "runs" / f"{bench}_main"


def status(bench):
    p = run_dir(bench) / "status.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def preflight_passed(bench):
    p = WORK / "advisor_cache" / "preflight" / f"{bench}.json"
    try:
        rep = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return bool(rep.get("passed")) and rep.get("advisor_mode") == "base"


def start(bench, arms):
    session = f"mcq_main_{bench}"
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    cmd = [PY, "-m", "src.manager.mcq_rsi", "run", "--config", f"configs/mcq_rsi_{bench}.json",
           "--run-dir", str(run_dir(bench)), "--hours", "72", "--advisor-url", URL]
    if arms:
        cmd += ["--arms", arms]
    logf = WORK / "logs" / f"{bench}_main.log"
    shell = f"cd {REPO} && {' '.join(cmd)} 2>&1 | tee -a {logf}; exec bash"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], env=ENV, check=True)
    log(f"started {bench} (arms {arms or 'config'}) in tmux {session}")


def final_exit_file(bench):
    return run_dir(bench) / "final_test_exit.txt"


def start_final(bench):
    session = f"mcq_final_{bench}"
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    logf = WORK / "logs" / f"{bench}_final_test.log"
    exitf = final_exit_file(bench)
    shell = (f"cd {REPO} && {PY} -m src.manager.mcq_rsi final-test --run-dir {run_dir(bench)} 2>&1 | tee -a {logf}; "
             f"echo ${{PIPESTATUS[0]}} > {exitf}; exec bash")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], env=ENV, check=True)
    log(f"started the locked final-test of {bench} in tmux {session} (log {logf})")


def parity_decision(bench, st):
    """(acknowledge?, reason) for a run stopped at a gate; None if it is not a round-1 parity stop."""
    stage = st.get("current_stage") or ""
    if stage != "r1/S1_dev":
        return None
    p = run_dir(bench) / "r1" / "S1_dev" / "parity.json"
    try:
        par = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if par.get("status") != "fail" or par.get("acknowledged"):
        return None
    failing = [c for c in par.get("checks", []) if c.get("gated") and not c.get("passed")]
    gaps = []
    for c in failing:
        if "accuracy" not in str(c.get("target", "")):
            return (False, f"gated check {c.get('target')} failed (not an accuracy check)")
        gaps += [abs(v) for v in (c.get("deltas") or {}).values() if isinstance(v, (int, float))]
    if not failing or not gaps:
        return (False, "parity failed without a measurable accuracy gap")
    gap = max(gaps)
    observed = {c.get("target"): c.get("observed") for c in par.get("checks", [])}
    if gap <= PARITY_MAX_ACC_GAP:
        return (True, f"auto-acknowledged overnight under the operator rule (accuracy gap {gap:.3f} <= "
                      f"{PARITY_MAX_ACC_GAP}, every other gated check passed): {json.dumps(observed)}; numeric "
                      f"drift from A100 + fla kernels + batched advisor serving, as for MedQA")
    return (False, f"accuracy gap {gap:.3f} > {PARITY_MAX_ACC_GAP}")


def ack(bench, reason):
    out = subprocess.run([PY, "-m", "src.manager.mcq_rsi", "ack-gate", "--run-dir", str(run_dir(bench)),
                          "--stage", "r1/S1_dev", "--reason", reason], cwd=REPO, env=ENV,
                         capture_output=True, text=True)
    log(f"ack {bench}: rc={out.returncode} {out.stdout.strip()[-300:]} {out.stderr.strip()[-300:]}")
    return out.returncode == 0


def load_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"queue": QUEUE, "active": {"medqa": {"arms": None, "restarts": 0}}, "done": {}}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=2))


def main():
    # The wrapper's preflight loop (one process for all three benchmarks; its tmux session outlives it).
    while subprocess.run(["pgrep", "-f", "runpod_mcq_rsi.sh preflight"], capture_output=True).returncode == 0:
        log("waiting for the mmlu_pro/gpqa/aqua preflight to finish")
        time.sleep(60)
    s = load_state()
    log(f"scheduler up: {json.dumps(s)}")
    while True:
        for bench in list(s["active"]):
            info = s["active"][bench]
            if info.get("kind") == "final":
                name = bench.split(":", 1)[1]
                exitf = final_exit_file(name)
                if exitf.is_file():
                    code = exitf.read_text().strip()
                    log(f"{bench}: final-test exited {code}")
                    s["done"][bench] = "completed" if code == "0" else f"final-test exit {code} (waiting for the operator)"
                    del s["active"][bench]
                elif subprocess.run(["tmux", "has-session", "-t", f"mcq_final_{name}"], capture_output=True).returncode:
                    log(f"{bench}: final-test session vanished without an exit code")
                    s["done"][bench] = "final-test vanished"
                    del s["active"][bench]
                continue
            st = status(bench)
            if st is None:
                if time.time() - info.get("started", time.time()) > 900:
                    log(f"{bench}: no status.json 15 min after start -> stopped")
                    s["done"][bench] = "no status"
                    del s["active"][bench]
                continue
            state = st.get("controller")
            dead = state == "running" and not alive(st.get("controller_pid"))
            if state == "completed":
                log(f"{bench}: completed")
                s["done"][bench] = "completed"
                del s["active"][bench]
            elif state in ("failed", "interrupted", "deadline") or dead:
                why = st.get("error") or ("controller died" if dead else state)
                decision = parity_decision(bench, st) if state == "failed" else None
                if decision is not None:
                    ok, reason = decision
                    if ok and ack(bench, reason):
                        start(bench, info.get("arms"))
                        info["started"] = time.time()
                        continue
                    log(f"{bench}: parity stop kept ({reason}) -> stopped, waiting for the operator")
                    s["done"][bench] = f"parity stop: {reason}"
                    del s["active"][bench]
                elif state == "deadline":
                    log(f"{bench}: deadline -> stopped")
                    s["done"][bench] = "deadline"
                    del s["active"][bench]
                elif info.get("restarts", 0) < 1:
                    info["restarts"] = info.get("restarts", 0) + 1
                    log(f"{bench}: {why} -> resuming once")
                    start(bench, info.get("arms"))
                    info["started"] = time.time()
                else:
                    log(f"{bench}: {why} again -> stopped, waiting for the operator")
                    s["done"][bench] = f"failed: {why}"
                    del s["active"][bench]
        while len(s["active"]) < MAX_CONCURRENT and s["queue"]:
            bench = s["queue"].pop(0)
            if bench.startswith("final:"):
                name = bench.split(":", 1)[1]
                if final_exit_file(name).exists():
                    log(f"{bench}: final-test already ran -> skipped")
                    continue
                start_final(name)
                s["active"][bench] = {"kind": "final", "started": time.time()}
                continue
            if not preflight_passed(bench):
                log(f"{bench}: no passing base-mode preflight -> skipped")
                s["done"][bench] = "no preflight"
                continue
            start(bench, ARMS)
            s["active"][bench] = {"arms": ARMS, "restarts": 0, "started": time.time()}
        save_state(s)
        if not s["active"] and not s["queue"]:
            log(f"nothing left to run: {json.dumps(s['done'])}")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
