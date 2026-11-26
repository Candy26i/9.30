#!/usr/bin/env python3
"""Job queue of the equal-information grid (scripts/mcq_verifier_grid.py) on one pod (rebuttal, 2026-11).

A job is (checkpoint name, benchmark, pool). Jobs run in the priority order of ``PHASES``, one manager process per GPU
slot (``CUDA_VISIBLE_DEVICES``), all against the one advisor server. A job is finished when its ``grid_summary.json``
exists; a job that exits without it is started again with ``--resume`` (its shards are kept), at most ``--retries``
more times. After every finished job the offline statistics of that (checkpoint, benchmark) are rewritten
(``mcq_verifier_grid_analyze.py``; a test pool uses the dev grid for the confidence threshold once that exists).

Checkpoints:  s1     the paper's one-round SFT manager          $WORK/import/<bench>/round1/sft
              dyn3   v3 dynamic round 3 (SFT only)              $CKPT/<bench>_dyn3
              dyn3g  MMLU-Pro v3 + FA-GRPO dynamic round 3      $CKPT/mmlu_pro_dyn3g
              base   the untrained base model as the manager    the path in $CKPT/base_model_path.txt

Slots: ``--slots 0:1,1:3`` (GPU:processes), or by default from the free memory of each GPU (one process per
``--mib-per-job``, at most ``--max-per-gpu``; start the advisor server first so that its share is already taken).

    python3 scripts/pod_ops/verifier_grid_queue.py                       # everything, in priority order
    python3 scripts/pod_ops/verifier_grid_queue.py --only '^s1/.*/test$'  # a subset
    python3 scripts/pod_ops/verifier_grid_queue.py --list                # print the jobs and exit

State: $WORK/runs/verifier_grid/queue_state.json; logs: $WORK/logs/verifier_grid/. Marker when the queue is empty:
$WORK/logs/verifier_grid/queue_done.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

WORK = Path(os.environ.get("MCQ_WORK", "/workspace/mcq_rsi"))
TREE = Path(os.environ.get("MCQ_TREE", "/workspace/9.30/agent_routing"))
PY = os.environ.get("MCQ_PY", "/workspace/mcq-venv/bin/python")
CKPT = Path(os.environ.get("VGRID_CKPT", str(WORK / "ckpt" / "verifier_grid")))
OUT = WORK / "runs" / "verifier_grid"
LOGS = WORK / "logs" / "verifier_grid"

BENCHES = ("medqa", "mmlu_pro", "aqua", "gpqa")  # longest pools first
# (checkpoint, pool, benchmarks) in priority order: the held-out comparison for the paper's manager first.
PHASES: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("s1", "test", BENCHES), ("s1", "dev", BENCHES),
    ("dyn3", "test", BENCHES), ("dyn3g", "test", ("mmlu_pro",)),
    ("dyn3", "dev", BENCHES), ("dyn3g", "dev", ("mmlu_pro",)),
    ("base", "test", BENCHES), ("base", "dev", BENCHES),
    ("s1", "test_paper", ("mmlu_pro",)),
)

Job = Tuple[str, str, str]  # checkpoint name, benchmark, pool


def job_id(job: Job) -> str:
    return "/".join(job)


def all_jobs(only: Optional[str] = None) -> List[Job]:
    jobs = [(ckpt, bench, pool) for ckpt, pool, benches in PHASES for bench in benches]
    if only:
        rx = re.compile(only)
        jobs = [j for j in jobs if rx.search(job_id(j))]
    return jobs


def checkpoint_path(ckpt: str, bench: str) -> str:
    if ckpt == "s1":
        return str(WORK / "import" / bench / "round1" / "sft")
    if ckpt == "base":
        return (CKPT / "base_model_path.txt").read_text(encoding="utf-8").strip()
    if ckpt in ("dyn3", "dyn3g"):
        return str(CKPT / f"{bench}_{ckpt}")
    raise ValueError(f"unknown checkpoint name {ckpt!r}")


def out_dir(job: Job) -> Path:
    ckpt, bench, pool = job
    return OUT / ckpt / bench / pool


def finished(job: Job) -> bool:
    return (out_dir(job) / "grid_summary.json").is_file()


def parse_slots(text: str) -> List[str]:
    """``"0:1,1:3"`` -> ["0", "1", "1", "1"]."""
    slots: List[str] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        gpu, sep, count = part.partition(":")
        if not gpu.isdigit() or (sep and not count.isdigit()):
            raise ValueError(f"bad slot spec {part!r} (expected GPU:COUNT)")
        slots += [gpu] * (int(count) if sep else 1)
    if not slots:
        raise ValueError("no slots")
    return slots


def auto_slots(free_mib: Sequence[int], mib_per_job: int = 26000, max_per_gpu: int = 3) -> List[str]:
    """One process per ``mib_per_job`` of free memory on each GPU, at most ``max_per_gpu``."""
    slots: List[str] = []
    for gpu, free in enumerate(free_mib):
        slots += [str(gpu)] * max(0, min(max_per_gpu, int(free) // int(mib_per_job)))
    return slots


def gpu_free_mib() -> List[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [int(x) for x in out.split()]


def command(job: Job) -> List[str]:
    ckpt, bench, pool = job
    out = out_dir(job)
    cmd = [PY, "scripts/mcq_verifier_grid.py", "--config", f"configs/mcq_rsi_{bench}_v3.json", "--pool", pool,
           "--checkpoint", checkpoint_path(ckpt, bench), "--out", str(out)]
    if (out / "grid_manifest.json").exists():
        cmd.append("--resume")
    return cmd


def launch(job: Job, gpu: str):
    """Start one grid process on ``gpu``; its output goes to the job's log."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / ("__".join(job) + ".log"), "ab")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONUNBUFFERED": "1", "MARGENT_WANDB_MODE": "disabled"}
    return subprocess.Popen(command(job), cwd=str(TREE), env=env, stdout=log, stderr=subprocess.STDOUT)


def analysis_commands(job: Job) -> List[List[str]]:
    """The analysis runs that the end of ``job`` makes possible or changes."""
    ckpt, bench, pool = job
    dev = out_dir((ckpt, bench, "dev")) / "grid_records.jsonl"
    targets = [pool] if pool != "dev" else ["dev"] + [p for p in ("test", "test_paper") if finished((ckpt, bench, p))]
    cmds = []
    for p in targets:
        cmd = [PY, "scripts/mcq_verifier_grid_analyze.py", "--records", str(out_dir((ckpt, bench, p)) / "grid_records.jsonl"),
               "--out", str(OUT / "analysis"), "--label", f"{ckpt}_{bench}_{p}"]
        if p != "dev" and dev.is_file():
            cmd += ["--dev", str(dev)]
        cmds.append(cmd)
    return cmds


def analyze(job: Job) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    for cmd in analysis_commands(job):
        with open(LOGS / "analysis.log", "ab") as log:
            rc = subprocess.run(cmd, cwd=str(TREE), env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        say(f"analysis {cmd[cmd.index('--label') + 1]}: {'ok' if rc == 0 else f'FAILED rc={rc}'}")


def say(msg: str) -> None:
    print(f"[vgrid-queue {time.strftime('%F %T')}] {msg}", flush=True)


def run_queue(jobs: Sequence[Job], slots: Sequence[str], *, launch: Callable = launch, finished: Callable = finished,
              after: Optional[Callable] = None, retries: int = 2, poll_seconds: float = 20.0,
              state_path: Optional[Path] = None, sleep: Callable = time.sleep) -> Dict[str, Dict[str, Any]]:
    """Run ``jobs`` in order on ``slots`` (a list of GPU ids, one entry per process)."""
    state: Dict[str, Dict[str, Any]] = {}
    pending: List[Job] = []
    for job in jobs:
        if finished(job):
            state[job_id(job)] = {"status": "done", "attempts": 0, "note": "finished earlier"}
        else:
            state[job_id(job)] = {"status": "pending", "attempts": 0}
            pending.append(job)
    free = list(slots)
    running: List[Tuple[Job, str, Any, float]] = []

    def save():
        if state_path is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
            tmp.replace(state_path)

    say(f"{len(pending)} jobs to run, {len(jobs) - len(pending)} finished earlier; slots {','.join(slots)}")
    save()
    while pending or running:
        for item in list(running):
            job, gpu, proc, started = item
            rc = proc.poll()
            if rc is None:
                continue
            running.remove(item)
            free.append(gpu)
            entry = state[job_id(job)]
            entry["seconds"] = round(entry.get("seconds", 0) + time.time() - started, 1)
            if finished(job):
                entry["status"] = "done"
                say(f"done {job_id(job)} (gpu {gpu}, {entry['seconds']:.0f}s, attempt {entry['attempts']})")
                if after is not None:
                    after(job)
            elif entry["attempts"] <= retries:
                entry["status"] = "pending"
                pending.insert(0, job)
                say(f"exit rc={rc} without a summary: {job_id(job)} (attempt {entry['attempts']}); it will resume")
            else:
                entry["status"] = "failed"
                say(f"FAILED {job_id(job)} after {entry['attempts']} attempts (rc={rc}); see its log")
            save()
        while pending and free:
            job, gpu = pending.pop(0), free.pop(0)
            entry = state[job_id(job)]
            entry.update(status="running", attempts=entry["attempts"] + 1, gpu=gpu)
            try:
                proc = launch(job, gpu)
            except Exception as exc:  # e.g. a checkpoint that was not fetched: the other jobs still run
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                free.append(gpu)
                say(f"FAILED to start {job_id(job)}: {entry['error']}")
                save()
                continue
            running.append((job, gpu, proc, time.time()))
            say(f"start {job_id(job)} on gpu {gpu} (attempt {entry['attempts']})")
            save()
        if running:
            sleep(poll_seconds)
    counts = {s: sum(e["status"] == s for e in state.values()) for s in ("done", "failed")}
    say(f"queue empty: {counts['done']} done, {counts['failed']} failed")
    return state


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="regular expression on checkpoint/benchmark/pool")
    ap.add_argument("--slots", default=None, help="GPU:COUNT,... (default: from the free memory of each GPU)")
    ap.add_argument("--mib-per-job", type=int, default=26000)
    ap.add_argument("--max-per-gpu", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    jobs = all_jobs(args.only)
    if args.list:
        for job in jobs:
            print(job_id(job), "done" if finished(job) else "pending")
        return 0
    slots = parse_slots(args.slots) if args.slots else auto_slots(gpu_free_mib(), args.mib_per_job, args.max_per_gpu)
    if not slots:
        raise SystemExit("no GPU has enough free memory for one manager process (see --mib-per-job)")
    state = run_queue(jobs, slots, after=analyze, retries=args.retries, state_path=OUT / "queue_state.json")
    failed = [k for k, e in state.items() if e["status"] != "done"]
    if not failed:
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / "queue_done").write_text(time.strftime("%F %T") + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
