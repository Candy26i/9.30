#!/usr/bin/env python3
"""Overnight supervisor (operator decisions of 2026-10-04): keep the remaining MCQ RSI work moving.

- RSI runs (main checkout, their own configs): resumed once after a failure; a second failure waits.
- RSI run completed -> its locked final-test starts from the worktree FINAL_TREE with --allow-code-change
  (lenient forced evals, floor 0.90; recorded in final/code_override.json).
- A final-test stopped at an analysis-only forced dev stage (final/<label>/dev_forced_*) -> that stage is moved
  aside (retry-stage) and the final-test resumes from FINAL_TREE with --allow-code-change (once per stage).
  A stop at a locked test stage waits for the operator.
Never edits a repository checkout. Log: /workspace/mcq_rsi/logs/supervisor.log
"""
import json
import os
import subprocess
import time
from pathlib import Path

WORK = Path("/workspace/mcq_rsi")
MAIN = Path("/workspace/9.30/agent_routing")
FINAL_TREE = Path("/workspace/9.30-final2/agent_routing")
PY = "/workspace/mcq-venv/bin/python"
URL = "http://127.0.0.1:18002"
ENV = {**os.environ, "HF_HOME": "/workspace/hf-cache", "HF_HUB_DISABLE_XET": "1", "TMPDIR": "/workspace/tmp",
       "MCQ_ADVISOR_PORT": "18002", "PYTHONUNBUFFERED": "1"}
ENV.pop("MARGENT_WANDB_MODE", None)
RSI = {  # bench -> how to resume its RSI run (None: RSI already complete)
    "gpqa": {"config": "/workspace/tmp/gpqa_main_cfg.json", "arms": "dynamic,static"},
    "aqua": {"config": "/workspace/tmp/aqua_main_cfg.json", "arms": "dynamic,static"},
    "medqa": None,
    "mmlu_pro": None,
}
REASON = ("forced (analysis) dev evals lenient (floor 0.90) under code {head}; locked test evals keep the hard gate. "
          "Operator-approved overnight automation 2026-10-04.")
STATE = WORK / "logs" / "supervisor_state.json"


def log(msg):
    print(f"[mcq-supervisor {time.strftime('%F %T')}] {msg}", flush=True)


def run_dir(b):
    return WORK / "runs" / f"{b}_main"


def status(b):
    try:
        return json.loads((run_dir(b) / "status.json").read_text())
    except (OSError, ValueError):
        return {}


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def tmux(session, shell):
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell], env=ENV, check=True)


def head(tree):
    return subprocess.run(["git", "-C", str(tree), "log", "--format=%h", "-1"], capture_output=True,
                          text=True).stdout.strip()


def start_final(b):
    reason = REASON.format(head=head(FINAL_TREE)).replace('"', "'")
    logf = WORK / "logs" / f"{b}_final_test.log"
    exitf = run_dir(b) / "final_test_exit.txt"
    exitf.unlink(missing_ok=True)
    shell = (f'cd {FINAL_TREE} && {PY} -m src.manager.mcq_rsi final-test --run-dir {run_dir(b)} '
             f'--allow-code-change "{reason}" 2>&1 | tee -a {logf}; echo ${{PIPESTATUS[0]}} > {exitf}; exec bash')
    tmux(f"mcq_final_{b}", shell)
    log(f"{b}: final-test started from {FINAL_TREE} ({head(FINAL_TREE)})")


def resume_rsi(b):
    spec = RSI[b]
    logf = WORK / "logs" / f"{b}_main.log"
    shell = (f"cd {MAIN} && {PY} -m src.manager.mcq_rsi run --config {spec['config']} --run-dir {run_dir(b)} "
             f"--hours 72 --advisor-url {URL} --arms {spec['arms']} 2>&1 | tee -a {logf}; exec bash")
    tmux(f"mcq_main_{b}", shell)
    log(f"{b}: RSI resumed")


def retry_stage(b, name):
    out = subprocess.run([PY, "-m", "src.manager.mcq_rsi", "retry-stage", "--run-dir", str(run_dir(b)),
                          "--name", name], cwd=FINAL_TREE, env=ENV, capture_output=True, text=True)
    log(f"{b}: retry-stage {name}: rc={out.returncode} {out.stdout.strip()[-200:]} {out.stderr.strip()[-300:]}")
    return out.returncode == 0


def load():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"rsi_resumes": {}, "forced_retries": [], "final_started": [], "done": {}}


def main():
    s = load()
    log(f"supervisor up: {json.dumps(s)}")
    while True:
        busy = False
        for b, spec in RSI.items():
            if b in s["done"]:
                continue
            st = status(b)
            ctl, stage = st.get("controller"), st.get("current_stage") or ""
            running = alive(st.get("controller_pid"))
            if ctl == "final-test complete":
                log(f"{b}: final-test complete")
                s["done"][b] = "final-test complete"
                continue
            busy = True
            if running:
                continue  # an RSI run or a final-test is in progress
            if ctl == "completed":  # RSI done, final-test not yet run (or refused before updating status)
                exitf = run_dir(b) / "final_test_exit.txt"
                if b not in s["final_started"]:
                    s["final_started"].append(b)
                    start_final(b)
                elif exitf.is_file() and exitf.read_text().strip() not in ("", "0"):
                    log(f"{b}: final-test exited {exitf.read_text().strip()} before starting -> waiting for the operator")
                    s["done"][b] = "final-test refused at start"
                continue
            if ctl in ("failed", "interrupted") and stage.startswith("final/"):
                leaf = stage.rsplit("/", 1)[-1]
                if leaf.startswith("dev_forced_") and stage not in s["forced_retries"]:
                    s["forced_retries"].append(stage)
                    if retry_stage(b, stage):
                        start_final(b)
                        continue
                log(f"{b}: final-test stopped at {stage} ({st.get('error')}) -> waiting for the operator")
                s["done"][b] = f"final-test stopped at {stage}"
                continue
            if ctl in ("failed", "interrupted") and spec is not None:
                n = s["rsi_resumes"].get(b, 0)
                if n < 1:
                    s["rsi_resumes"][b] = n + 1
                    log(f"{b}: RSI {ctl} at {stage} ({st.get('error')}) -> resuming once")
                    resume_rsi(b)
                    continue
                log(f"{b}: RSI {ctl} again at {stage} -> waiting for the operator")
                s["done"][b] = f"RSI stopped at {stage}"
                continue
            if ctl == "final-test" and not running:  # final-test process died without a status update
                if b not in s["forced_retries"]:
                    s["forced_retries"].append(b)
                    log(f"{b}: final-test controller died at {stage} -> resuming the final-test")
                    start_final(b)
                    continue
                log(f"{b}: final-test controller died again -> waiting for the operator")
                s["done"][b] = "final-test died"
        STATE.write_text(json.dumps(s, indent=2))
        if not busy:
            log(f"all done: {json.dumps(s['done'])}")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
