"""Local, append-only experiment telemetry. No external logging service required."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import importlib.metadata
import os
import platform
from pathlib import Path
import socket
import shutil
import subprocess
import threading
import time
import traceback
import uuid

from ..utils.io import append_jsonl
from .wandb_tracking import WandbTracker, scalar_metrics, _config

ACTIVE = None


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Monitor:
    """Start only AFTER validating/creating the experiment manifest."""
    def __init__(self, output, stage, interval=20):
        self.root, self.stage, self.interval = Path(output), stage, interval
        self.attempt = uuid.uuid4().hex
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.tracker = WandbTracker(self.root, stage, self.attempt)
        self.wandb_failed = False
        self.had_wandb_error = False
        self.retry_after = 0.0
        self.totals = {}
        self.generation_count = 0
        self.last_artifact_flush = time.monotonic()
        self.artifact_interval = max(30, int(os.environ.get("MARGENT_WANDB_ARTIFACT_INTERVAL", "300")))
        self.question = {}
        self.state = {"schema_version": 1, "stage": stage, "attempt": self.attempt,
                      "status": "running", "started_at": now(), "pid": os.getpid(),
                      "hostname": socket.gethostname(), "progress": {}}

    def __enter__(self):
        global ACTIVE
        self.root.mkdir(parents=True, exist_ok=True)
        events = self.root / "events.jsonl"
        if events.exists():
            for line in events.read_text().splitlines():
                try:
                    if json.loads(line).get("event") == "usage_tail_recovered":
                        self.state["usage_incomplete"] = True
                except json.JSONDecodeError:
                    self.state["usage_incomplete"] = True
        journal = self.root / "generations.jsonl"
        if journal.exists():
            with journal.open() as records:
                self.generation_count = sum(bool(line.strip()) for line in records)
        ledger = self.root / "usage.jsonl"
        if ledger.exists():
            lines = ledger.read_bytes().splitlines(keepends=True)
            offset = 0
            for index, line in enumerate(lines):
                try:
                    if line.strip():
                        self._count_usage(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if index != len(lines) - 1 or line.endswith(b"\n"):
                        raise  # Interior corruption is not a safe tail recovery.
                    (self.root / f"usage_torn_tail_{self.attempt}.txt").write_bytes(line)
                    with ledger.open("r+b") as out:
                        out.truncate(offset)
                    self.event("usage_tail_recovered", discarded_bytes=len(line),
                               note="Incomplete usage cannot be reconstructed; token totals are lower bounds")
                    self.state["usage_incomplete"] = True
                    break
                offset += len(line)
            else:
                if lines and not lines[-1].endswith(b"\n"):
                    with ledger.open("ab") as out:
                        out.write(b"\n")
        try:
            self.tracker.start()
        except Exception:
            try:
                self.tracker.finish("failed")
            except Exception:
                pass
            raise
        self.previous, ACTIVE = ACTIVE, self
        self.started = time.monotonic()
        packages = {}
        for name in ("torch", "transformers", "trl", "peft", "datasets", "math-verify", "matplotlib", "wandb"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        try:
            git = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            commit = git.stdout.strip() if git.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            commit = None
        environment = {"packages": packages, "git_commit": commit,
                    "python": platform.python_version(), "platform": platform.platform(),
                    "hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "note": "GPU samples are device-level and can include other processes"}
        atomic_json(self.root / f"environment_{self.attempt}.json", environment)
        self.summary({"environment": environment})
        self.event("started")
        self.update()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def update(self, **progress):
        with self.lock:
            if progress:
                self.state["last_progress_at"] = now()
            self.state["progress"].update(progress)
            self.state.update(updated_at=now(), elapsed_seconds=time.monotonic() - self.started,
                              disk_free_bytes=shutil.disk_usage(self.root).free)
            atomic_json(self.root / "status.json", self.state)

    def event(self, event, **values):
        with self.lock:
            append_jsonl(str(self.root / "events.jsonl"), [{"time": now(), "attempt": self.attempt,
                         "event": event, **values}])

    def usage(self, role, value, **extra):
        fields = ("prompt_tokens", "completion_tokens", "actual_prompt_tokens", "actual_completion_tokens",
                  "seconds", "cache_hit", "truncated", "finish_reason", "valid_output", "error")
        item = {k: value[k] for k in fields if k in value}
        item.update(time=now(), attempt=self.attempt, role=role, context=dict(self.state["progress"]), **extra)
        with self.lock:
            append_jsonl(str(self.root / "usage.jsonl"), [item])
            self._count_usage(item)

    def _count_usage(self, item):
        prefix = "usage/" + item["role"] + "/"
        values = {"prompt_tokens": item.get("actual_prompt_tokens", item.get("prompt_tokens", 0)),
                  "generated_tokens": item.get("actual_completion_tokens", item.get("completion_tokens", 0)),
                  "seconds": item.get("seconds", 0), "cache_hits": int(item.get("cache_hit", False)),
                  "input_tokens": item.get("input_tokens", 0),
                  "supervised_tokens": item.get("supervised_tokens", 0)}
        for key, value in values.items():
            self.totals[prefix + key] = self.totals.get(prefix + key, 0) + value

    def log_wandb(self, values):
        if time.monotonic() < self.retry_after:
            return
        try:
            self.tracker.log(values)
            if self.wandb_failed:
                self.retry_after = 0.0
                self.state["wandb_status"] = "recovered_after_upload_error"
                self.event("wandb_recovered")
                self.wandb_failed = False
        except Exception as exc:
            self.wandb_warning(exc)

    def wandb_warning(self, exc):
        self.wandb_failed = True
        self.had_wandb_error = True
        self.retry_after = time.monotonic() + 30
        self.state["wandb_status"] = "upload_error_local_logs_retained"
        self.event("wandb_warning", error_type=type(exc).__name__)
        print("[wandb] Upload failed; computation continues with local logs. Check events.jsonl.", flush=True)

    def summary(self, values):
        """Persist descriptive metadata locally and remotely without losing strings."""
        with self.lock:
            path = self.root / "run_summary.json"
            current = json.loads(path.read_text()) if path.exists() else {}
            current.update(_config(values))
            atomic_json(path, current)
            if time.monotonic() < self.retry_after:
                return
            try:
                self.tracker.summary(current)
            except Exception as exc:
                self.wandb_warning(exc)

    def snapshot_artifacts(self):
        if time.monotonic() < self.retry_after:
            return
        with self.lock:
            try:
                self.tracker.snapshot_artifacts()
                self.last_artifact_flush = time.monotonic()
            except Exception as exc:
                self.wandb_warning(exc)

    def set_question(self, row):
        from .data import identity
        self.question = {"question_hash": identity(row.question), "question": row.question,
                         "context": row.context, "ground_truth": row.ground_truth,
                         "benchmark_name": row.benchmark_name, "split": row.split,
                         "example_id": row.example_id}

    def generation(self, role, value, messages=None, **extra):
        fields = ("text", "valid", "truncated", "prompt_tokens", "completion_tokens",
                  "actual_prompt_tokens", "actual_completion_tokens", "seconds", "cache_hit", "error", "finish_reason", "valid_output",
                  "requested_max_tokens", "max_context", "actual_role", "advisor_role")
        record = {k: value[k] for k in fields if k in value}
        record.update(self.question, source="live", attempt=self.attempt, role=role, time=now(),
                      phase=self.state["progress"].get("phase"),
                      sequence=list(self.state["progress"].get("sequence", [])), messages=messages)
        record.update({k: extra[k] for k in ("advisor", "operation", "max_tokens", "error") if k in extra})
        with self.lock:
            # Save before the caller can raise for truncation or fail the question.
            append_jsonl(str(self.root / "generations.jsonl"), [record])
            self.log_text("generation", record)
            self.generation_count += 1
            prefix = role + ("/" + extra["advisor"] if extra.get("advisor") else "")
            self.metrics({prefix + "/truncated": int(bool(record.get("truncated"))),
                          prefix + "/seconds": record.get("seconds", 0),
                          prefix + "/completion_tokens": record.get("actual_completion_tokens", record.get("completion_tokens", 0))},
                         "generation", generation_step=self.generation_count)

    def log_text(self, kind, record):
        if time.monotonic() < self.retry_after:
            return
        with self.lock:
            try:
                self.tracker.log_text(kind, record)
            except Exception as exc:
                self.wandb_warning(exc)

    def completed_question(self, record, source="live"):
        self.log_text("question", {**record, "source": source})
        self.flush_tables(force=True)

    def flush_tables(self, force=False):
        if time.monotonic() < self.retry_after:
            return
        with self.lock:
            try:
                self.tracker.flush_tables(force=force)
            except Exception as exc:
                self.wandb_warning(exc)

    def metrics(self, values, namespace="", **axes):
        values = {**scalar_metrics(values, namespace), **scalar_metrics(axes)}
        with self.lock:
            append_jsonl(str(self.root / "metrics.jsonl"), [{"time": now(), "attempt": self.attempt, **values}])
            self.log_wandb(values)

    def _heartbeat(self):
        while not self.stop.is_set():
            try:
                self.update()
                with self.lock:
                    self.summary({"heartbeat_at": self.state["updated_at"],
                                  "last_progress_at": self.state.get("last_progress_at"),
                                  "progress": self.state["progress"], "attempt": self.attempt})
                    self.log_wandb({"system/elapsed_seconds": self.state["elapsed_seconds"],
                                    "system/disk_free_gib": self.state["disk_free_bytes"] / 2 ** 30,
                                    **scalar_metrics(self.state["progress"], "progress"), **self.totals})
                    self.flush_tables()
                    if time.monotonic() - self.last_artifact_flush >= self.artifact_interval:
                        self.snapshot_artifacts()
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,power.draw",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    with self.lock:
                        append_jsonl(str(self.root / "gpu_samples.jsonl"), [{"time": now(), "attempt": self.attempt,
                            "columns": ["index", "uuid", "memory_used_mib", "memory_total_mib", "utilization_percent", "power_watts"],
                            "devices": [line.split(", ") for line in result.stdout.strip().splitlines()]}])
                        gpu = {}
                        for line in result.stdout.strip().splitlines():
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) != 6:
                                continue
                            for key, val in zip(("memory_used_mib", "memory_total_mib", "utilization_percent", "power_watts"), parts[2:]):
                                try:
                                    gpu[f"gpu/{parts[0]}/{key}"] = float(val)
                                except ValueError:
                                    pass
                        self.log_wandb(gpu)
            except FileNotFoundError:
                pass  # CPU validation has no nvidia-smi.
            except Exception as exc:
                self.event("monitor_warning", error=str(exc))
            self.stop.wait(self.interval)

    def __exit__(self, typ, value, tb):
        global ACTIVE
        self.stop.set()
        self.thread.join(timeout=6)
        elapsed = time.monotonic() - self.started
        status = "completed" if typ is None else "interrupted" if issubclass(typ, KeyboardInterrupt) else "failed"
        self.state.update(status=status, finished_at=now())
        if typ:
            error = "".join(traceback.format_exception(typ, value, tb))
            self.state["error"] = str(value)
            with (self.root / "errors.log").open("a", encoding="utf-8") as out:
                out.write(f"\n{now()} attempt={self.attempt}\n{error}")
        self.event(status, wall_seconds=elapsed, error=str(value) if typ else None)
        self.update()
        try:
            # A transient table/network error must not suppress final evidence.
            self.retry_after = 0.0
            self.summary({"status": status, "finished_at": self.state["finished_at"],
                          "progress": self.state["progress"], "error": str(value) if typ else None,
                          "error_type": typ.__name__ if typ else None,
                          "had_tracking_error": self.had_wandb_error,
                          "usage_incomplete": self.state.get("usage_incomplete", False)})
            self.log_wandb({**self.totals, "system/elapsed_seconds": elapsed})
            self.flush_tables(force=True)
            self.tracker.finish(status)
            self.wandb_failed = False
            if self.had_wandb_error:
                self.state["wandb_status"] = "recovered_with_local_evidence"
        except Exception as exc:
            self.wandb_warning(exc)
        finally:
            self.update()
            ACTIVE = self.previous


def progress(**values):
    if ACTIVE:
        ACTIVE.update(**values)


def usage(role, value, **extra):
    if ACTIVE:
        ACTIVE.usage(role, value, **extra)


def question_context(row):
    if ACTIVE:
        ACTIVE.set_question(row)


def generation(role, value, messages=None, **extra):
    if ACTIVE:
        ACTIVE.generation(role, value, messages, **extra)


def completed_question(record):
    if ACTIVE:
        ACTIVE.completed_question(record)


def rollout(record):
    if ACTIVE:
        with ACTIVE.lock:
            append_jsonl(str(ACTIVE.root / "rollout_diagnostics.jsonl"), [record])
            ACTIVE.log_text("rollout", record)


def metrics(values, namespace="", **axes):
    if ACTIVE:
        ACTIVE.metrics(values, namespace, **axes)


def summary(values):
    if ACTIVE:
        ACTIVE.summary(values)


def status_snapshot(run_dir):
    root = Path(run_dir)
    entries = []
    for path in sorted(root.rglob("status.json")):
        if {"tracking_exports", "wandb"} & set(path.relative_to(root).parts):
            continue
        state = json.loads(path.read_text())
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(state["updated_at"])).total_seconds()
        state["heartbeat_age_seconds"] = round(age, 1)
        if state.get("last_progress_at"):
            state["progress_age_seconds"] = round((datetime.now(timezone.utc) - datetime.fromisoformat(state["last_progress_at"])).total_seconds(), 1)
        if state["status"] == "running":
            state["liveness"] = "recent_heartbeat" if age < 120 else "stale_heartbeat"
            if state["hostname"] == socket.gethostname():
                try:
                    os.kill(state["pid"], 0)
                except ProcessLookupError:
                    state["liveness"] = "process_missing"
                except PermissionError:
                    state["liveness"] = "process_not_accessible"
        entries.append({"path": str(path.parent.relative_to(root)), **state})
    plan = json.loads((root / "loop.json").read_text()).get("plan", []) if (root / "loop.json").exists() else []
    return {"run_dir": str(root), "states": entries,
            "stages": [{"stage": p["stage"], "path": p["output"],
                        "complete": (Path(p["output"]) / ".stage_complete.json").exists()} for p in plan],
            "note": "Heartbeat reports process activity, not convergence. SIGKILL/power loss is detected as stale or missing."}


def training_callback(output):
    from transformers import TrainerCallback

    class Callback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            progress(phase="training", step=state.global_step, max_steps=state.max_steps)

        def on_log(self, args, state, control, logs=None, **kwargs):
            append_jsonl(str(Path(output) / "training_log.jsonl"), [{"time": now(),
                         "attempt": ACTIVE.attempt if ACTIVE else None, "step": state.global_step,
                         "epoch": state.epoch, **(logs or {})}])
            metrics(logs or {}, "train", trainer_step=state.global_step)

        def on_step_end(self, args, state, control, **kwargs):
            progress(phase="training", step=state.global_step, max_steps=state.max_steps, epoch=state.epoch)

        def on_save(self, args, state, control, **kwargs):
            if ACTIVE:
                ACTIVE.event("checkpoint_saved", step=state.global_step,
                             path=str(Path(output) / f"checkpoint-{state.global_step}"))

    return Callback()
