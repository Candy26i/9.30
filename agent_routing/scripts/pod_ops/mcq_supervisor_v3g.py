#!/usr/bin/env python3
"""Supervisor for the 2026-10-09 pod: the v3 setting WITH GRPO (arms dynamic, static, success) on four benchmarks.

- ``<bench>_v3g``: arms dynamic, static, success from configs/mcq_rsi_<bench>_v3.json (draft_supervision none) with the
  per-round FA-GRPO of the paper rule, one run per GPU at a time (GPU 0 also holds the vLLM advisors; GPQA's 16k-token SFT
  needs a card to itself). User decision 2026-10-09: the method is SFT + GRPO, so the fixed-loss grid must include GRPO.
- Both GPUs are kept for experiment D (``logs/grpoq_done``) and the label-quality chains (``logs/labelq_done_gpu0/1``) first.
- Acknowledgements (user delegation 2026-10-07): right after a run's ``rsi_run.json`` appears, ``r1/S1_dev`` (round-1
  parity, acknowledged for every version so far) and the validity-only rule for the S_k dev evals and locked tests.
- Resume once per run; final-test with ``--reuse-test``; state in ``logs/supervisor_v3g_state.json``.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/workspace/mcq_rsi")
TREE = Path("/workspace/9.30/agent_routing")
PY = "/workspace/mcq-venv/bin/python"
URL = "http://127.0.0.1:18002"
ENVLINE = "HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_XET=1 TMPDIR=/workspace/tmp MCQ_ADVISOR_PORT=18002 PYTHONUNBUFFERED=1"
ENV = {**os.environ, "HF_HOME": "/workspace/hf-cache", "HF_HUB_DISABLE_XET": "1", "TMPDIR": "/workspace/tmp",
       "MCQ_ADVISOR_PORT": "18002", "PYTHONUNBUFFERED": "1"}
ENV.pop("MARGENT_WANDB_MODE", None)
ARMS = "dynamic,static,success"
RUNS = [{"name": "medqa_v3g", "bench": "medqa", "gpu": "0"}, {"name": "mmlu_pro_v3g", "bench": "mmlu_pro", "gpu": "0"},
        {"name": "gpqa_v3g", "bench": "gpqa", "gpu": "1"}, {"name": "aqua_v3g", "bench": "aqua", "gpu": "1"}]
# User order 2026-10-09: experiment D and the label-quality chains first, the GRPO grid last.
GPU_MARKERS = {"0": (WORK / "logs" / "labelq_done_gpu0",), "1": (WORK / "logs" / "grpoq_done", WORK / "logs" / "labelq_done_gpu1")}
STATE = WORK / "logs" / "supervisor_v3g_state.json"
ACK_REASON = ("Pre-acknowledged per the user delegation of 2026-10-07: only a validity-rate gate failure (unparsed answers, "
              "valid >= ACK_MIN_VALID 0.99) may pass; any other failure still stops the run.")
PARITY_REASON = ("Round-1 parity acknowledged as for the v1/v2/v3 runs (MedQA S_1 is the substitute checkpoint; AQuA dev "
                 "0.73 vs the paper's 0.78); the locked test of S_1 reproduced the paper. User delegation 2026-10-07.")


def log(msg):
    print(f"[mcq-supervisor-v3g {time.strftime('%F %T')}] {msg}", flush=True)


def status(r):
    try:
        return json.loads((WORK / "runs" / r / "status.json").read_text())
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


def write_cfg(run):
    cfg = json.loads((TREE / "configs" / f"mcq_rsi_{run['bench']}_v3.json").read_text())
    cfg["gpus"] = {"inference": run["gpu"], "train": run["gpu"]}
    path = Path("/workspace/tmp") / f"{run['name']}_cfg.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def start_run(run):
    cfg = write_cfg(run)
    logf = WORK / "logs" / f"{run['name']}.log"
    shell = (f"cd {TREE} && {ENVLINE} {PY} -m src.manager.mcq_rsi run --config {cfg} --run-dir {WORK / 'runs' / run['name']} "
             f"--hours 72 --advisor-url {URL} --arms {ARMS} 2>&1 | tee -a {logf}; exec bash")
    tmux(f"mcq_{run['name']}", shell)
    log(f"{run['name']}: RSI (re)started on GPU {run['gpu']} from {TREE}")


def start_final(run):
    logf = WORK / "logs" / f"{run['name']}_final_test.log"
    exitf = WORK / "runs" / run["name"] / "final_test_exit.txt"
    exitf.unlink(missing_ok=True)
    reason = (f"v3 setting with GRPO (draft_supervision none; arms dynamic, static, success); {run['bench']}_main / _v3 "
              f"already ran the locked test for S_1. User decision 2026-10-09.")
    shell = (f'cd {TREE} && {ENVLINE} {PY} -m src.manager.mcq_rsi final-test --run-dir {WORK / "runs" / run["name"]} '
             f'--reuse-test "{reason}" 2>&1 | tee -a {logf}; echo ${{PIPESTATUS[0]}} > {exitf}; exec bash')
    tmux(f"mcq_final_{run['name']}", shell)
    log(f"{run['name']}: final-test started (--reuse-test)")


def record_acks(run):
    """Once rsi_run.json exists: parity + validity acknowledgements (idempotent)."""
    root = WORK / "runs" / run["name"]
    if not (root / "rsi_run.json").is_file():
        return False
    arms = ("dynamic", "static", "success")
    stages = [("r1/S1_dev", PARITY_REASON)] + [(f"r{k}/{a}/sft_dev", ACK_REASON) for k in (2, 3) for a in arms]
    stages += [(f"final/{lab}/test", ACK_REASON) for lab in ("S_1",) + arms]
    if run["bench"] == "mmlu_pro":
        stages += [(f"final/{lab}/test_paper", ACK_REASON) for lab in ("S_1",) + arms]
    for stage, reason in stages:
        res = subprocess.run([PY, "-m", "src.manager.mcq_rsi", "ack-gate", "--run-dir", str(root), "--stage", stage,
                              "--reason", reason], cwd=str(TREE), env=ENV, capture_output=True, text=True)
        if res.returncode != 0:
            log(f"{run['name']}: ack-gate {stage} failed: {res.stderr.strip()[-300:]}")
            return False
    log(f"{run['name']}: {len(stages)} acknowledgements recorded")
    return True


def load():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {"started": [], "acked": [], "resumes": {}, "final_started": [], "final_resumes": {}, "done": {},
                "started_unix": {}, "held": []}


def main():
    only = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else {r["name"] for r in RUNS}
    s = load()
    log(f"supervisor_v3g up: runs {sorted(only)} state {json.dumps(s)}")
    boot = WORK / "logs" / "bootstrap.log"
    while not (boot.is_file() and "bootstrap complete" in boot.read_text()):
        time.sleep(60)  # the pod bootstrap (advisors, cache, preflight) must finish before any run starts
    while True:
        busy = False
        gpu_busy = {}
        for gpu, markers in GPU_MARKERS.items():
            if not all(m.exists() for m in markers):
                gpu_busy[gpu] = "labelq/grpoq"
        for run in RUNS:
            r = run["name"]
            if r not in only or r in s["done"]:
                continue
            busy = True
            if r not in s["started"]:
                continue
            if r not in s["acked"] and record_acks(run):
                s["acked"].append(r)
            st = status(r)
            if alive(st.get("controller_pid")):
                gpu_busy[run["gpu"]] = r
        for run in RUNS:
            r = run["name"]
            if r not in only or r in s["done"]:
                continue
            if r not in s["started"]:
                if run["gpu"] in gpu_busy:
                    continue
                s["started"].append(r); s["started_unix"][r] = time.time()
                start_run(run); gpu_busy[run["gpu"]] = r
                continue
            if r in s.get("held", []):
                if run["gpu"] in gpu_busy:
                    continue
                s["held"].remove(r); start_run(run); gpu_busy[run["gpu"]] = r
                log(f"{r}: GPU {run['gpu']} free -> continuing")
                continue
            st = status(r)
            ctl, stage = st.get("controller"), st.get("current_stage") or ""
            running = alive(st.get("controller_pid"))
            exitf = WORK / "runs" / r / "final_test_exit.txt"
            if ctl == "final-test complete":
                log(f"{r}: final-test complete"); s["done"][r] = "final-test complete"; continue
            if running:
                continue
            if not st:
                if time.time() - s["started_unix"].get(r, 0) > 300:
                    log(f"{r}: no status.json 5 min after the start -> waiting for the operator (logs/{r}.log)")
                    s["done"][r] = "never started"
                continue
            if ctl == "completed":
                if r not in s["final_started"]:
                    s["final_started"].append(r); start_final(run); gpu_busy[run["gpu"]] = r
                elif exitf.is_file() and exitf.read_text().strip() not in ("", "0"):
                    log(f"{r}: final-test exited {exitf.read_text().strip()} before starting -> waiting for the operator")
                    s["done"][r] = "final-test refused"
                continue
            if ctl in ("failed", "interrupted", "final-test") and stage.startswith("final/"):
                if s["final_resumes"].get(r, 0) < 1:
                    s["final_resumes"][r] = 1
                    log(f"{r}: final-test {ctl} at {stage} ({st.get('error')}) -> resuming once"); start_final(run)
                else:
                    log(f"{r}: final-test stopped again at {stage} -> waiting for the operator")
                    s["done"][r] = f"final-test stopped at {stage}"
                continue
            if ctl in ("failed", "interrupted", "running", "deadline"):
                if s["resumes"].get(r, 0) < 1:
                    if run["gpu"] in gpu_busy:
                        continue
                    s["resumes"][r] = 1
                    log(f"{r}: RSI {ctl} at {stage} ({st.get('error')}) -> resuming once"); start_run(run); gpu_busy[run["gpu"]] = r
                else:
                    log(f"{r}: RSI {ctl} again at {stage} ({st.get('error')}) -> waiting for the operator")
                    s["done"][r] = f"RSI stopped at {stage}"
        STATE.write_text(json.dumps(s, indent=2))
        if not busy:
            log(f"all done: {json.dumps(s['done'])}"); return
        time.sleep(60)


if __name__ == "__main__":
    main()
