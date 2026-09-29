"""Optional W&B metrics and explicitly enabled text tables with stage identities.

Enable explicitly with MARGENT_WANDB_MODE=online or offline. Local experiment
records remain authoritative. MARGENT_WANDB_TEXT=1 enables output tables;
model weights are never uploaded by this module.
"""
from __future__ import annotations

from importlib import import_module
import json
import hashlib
import math
import os
from pathlib import Path
import re
import time
import uuid


def tracking_mode():
    mode = os.environ.get("MARGENT_WANDB_MODE", "disabled").lower()
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("MARGENT_WANDB_MODE must be online, offline, or disabled")
    return mode


def text_tracking_enabled():
    return os.environ.get("MARGENT_WANDB_TEXT", "0").lower() in {"1", "true", "yes"}


def scalar_metrics(values, prefix=""):
    """Flatten finite scalars only. Text, examples, labels and lists stay local."""
    out = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            out[name] = value
    return out


def _read(path):
    return json.loads(path.read_text())


def _write(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def _config(value):
    """Do not send authentication fields if future experiment configs add them."""
    if isinstance(value, dict):
        return {k: "[redacted]" if re.search(r"api.?key|secret|password|credential|access.?token|authorization", k, re.I)
                else _config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_config(v) for v in value]
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def redact_text(value):
    """Scrub common credential representations from explicitly selected logs."""
    value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", value)
    value = re.sub(r"(?i)((?:[\w-]*(?:api[_-]?key|secret|password|access[_-]?token|authorization))[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,}]+",
                   r"\1[redacted]", value)
    return re.sub(r"(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", value)


class WandbTracker:
    def __init__(self, root, stage, attempt):
        self.run = None
        self.root = Path(root).resolve()
        self.stage, self.attempt = stage, attempt
        self.mode = tracking_mode()
        self.text_enabled = text_tracking_enabled()
        self.tables, self.table_counts, self.pending_tables = {}, {}, set()
        self.last_table_flush = None
        self.identity = None

    def start(self):
        if self.mode == "disabled":
            return
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise RuntimeError("W&B requested but not installed; install requirements-math.txt") from exc
        self.sdk = wandb
        if self.text_enabled:
            self.table_max_rows = int(os.environ.get("MARGENT_WANDB_TABLE_MAX_ROWS", "10000"))
            self.table_max_chars = int(os.environ.get("MARGENT_WANDB_TABLE_MAX_CHARS", "20000"))
            if self.table_max_rows < 1 or self.table_max_chars < 256:
                raise ValueError("W&B table limits require MAX_ROWS >= 1 and MAX_CHARS >= 256")
        project = os.environ.get("WANDB_PROJECT", "margent-math-rsi")
        entity = os.environ.get("WANDB_ENTITY")
        if not entity:
            raise ValueError("Set WANDB_ENTITY to your W&B username or team before enabling tracking")
        # A loop owns the group; its subprocess stages discover the same manifest.
        experiment = next((p for p in (self.root, *self.root.parents)
                           if any((p / name).exists() for name in ("expert_run.json", "rsi_run.json", "loop.json", "benchmark_run.json"))), self.root)
        manifest = next((experiment / name for name in ("expert_run.json", "rsi_run.json", "loop.json", "benchmark_run.json", "run.json", "training_run.json")
                         if (experiment / name).exists()), None)
        metadata = _read(manifest) if manifest else {}
        stage_manifest = next((self.root / name for name in ("training_run.json", "run.json")
                               if (self.root / name).is_file()), None)
        stage_metadata = _read(stage_manifest) if stage_manifest else metadata
        group_file = experiment / "wandb_experiment.json"
        if group_file.exists():
            group = _read(group_file)
            if (group["project"], group["entity"]) != (project, entity):
                raise ValueError("W&B project/entity changed for this experiment; use the original settings")
        else:
            group = {"project": project, "entity": entity,
                     "group": experiment.name + "-" + uuid.uuid4().hex[:8]}
            _write(group_file, group)
        identity_file = self.root / "wandb_run.json"
        identity = _read(identity_file) if identity_file.exists() else {
            **group, "stage": self.stage, "id": uuid.uuid4().hex[:12]}
        if any(identity[k] != v for k, v in {**group, "stage": self.stage}.items()):
            raise ValueError("W&B stage identity changed; use the original experiment directory")
        _write(identity_file, identity)
        self.identity = identity
        # Offline W&B cannot resume. Give each segment its own ID, retaining a
        # common logical_stage_id/group so later sync does not overwrite data.
        run_id = identity["id"] if self.mode == "online" else identity["id"] + "-" + self.attempt[:8]
        relative = self.root.relative_to(experiment).as_posix()
        name = group["group"] + "/" + (relative if relative != "." else self.stage)
        match = re.search(r"(?:^|/)round_(\d+)(?:/|$)", relative)
        config = _config(metadata.get("config", {}))
        config.update(_config(stage_metadata.get("config", {})))
        arm = metadata.get("arm", "standalone")
        if (experiment / "rsi_run.json").is_file():
            part = relative.split("/", 1)[0]
            arm = part if part in metadata.get("arms", ["dynamic", "static", "success"]) else "shared"
        config.update(arm=arm, stage=self.stage,
                      stage_path=relative, round=int(match[1]) if match else 0,
                      logical_stage_id=identity["id"],
                      experiment_manifest=_config(metadata), stage_manifest=_config(stage_metadata),
                      telemetry_schema_version=2)
        if stage_metadata.get('role'):
            config['expert_role'] = stage_metadata['role']
        self.run = wandb.init(project=project, entity=entity, group=group["group"],
            id=run_id, name=name, job_type=self.stage, config=config, mode=self.mode,
            **({"resume": "allow"} if self.mode == "online" else {}), dir=str(self.root),
            settings=wandb.Settings(init_timeout=60, console="off", disable_git=True, save_code=False))
        self.run.define_metric("trainer_step")
        self.run.define_metric("train/*", step_metric="trainer_step")
        self.run.define_metric("grpo/*", step_metric="trainer_step")
        self.run.define_metric("diagnostic_step")
        for pattern in ("eval/*", "internalization/*", "delegation/*"):
            self.run.define_metric(pattern, step_metric="diagnostic_step")
        self.run.define_metric("generation_step")
        self.run.define_metric("generation/*", step_metric="generation_step")
        self.run.summary.update({"attempt": self.attempt, "status": "running"})
        self.run.summary.update({"debug/text_logging": self.text_enabled})
        url = self.run.url if self.mode == "online" else None
        link = {**identity, "active_id": run_id, "mode": self.mode, "url": url,
                "attempt": self.attempt}
        _write(self.root / "wandb_link.json", link)
        print(f"[wandb] {name}: {url or 'offline records in ' + str(self.root / 'wandb')}", flush=True)

    def log(self, values):
        if self.run:
            clean = scalar_metrics(values)
            if clean:
                # W&B owns its monotonic history step; trainer_step can restart
                # from the last saved checkpoint without dropping new records.
                self.run.log(clean)

    def summary(self, values):
        if self.run:
            self.run.summary.update(_config(values))

    def snapshot_artifacts(self):
        """Persist bounded, allowlisted evidence; never discover/upload weights or credentials.

        Metadata/metrics/errors are always included when tracking is enabled.
        Question text and raw trajectories require the same explicit text opt-in
        as Tables. Table display limits never truncate this artifact's files.
        """
        if self.run is None:
            return
        names = {"run.json", "training_run.json", "loop.json", "rsi_run.json", "benchmark_run.json", "expert_run.json",
                 "expert_status.json", "expert_report.json", "expert_data_report.json", "experts.json", "manager_config.json",
                 "data_report.json", "dev_metrics.json", "expert_checkpoint.json", "expert_eval_report.json",
                 "summary.json", "run_summary.json", "status.json", "training_metrics.json",
                 "sft_data_report.json", "loop_report.json", "rsi_report.json", "report.json",
                 "pilot_report.json", "pilot_timeline.csv", "initial_gate.json", "config.json", "advisor_identity.json",
                 "frozen_advisor.json", "gpu_preflight.json", "checkpoint_inventory.json", "resume.json",
                 "baseline_status.json", "preflight_report.json", "baseline_report.json",
                 "resume_check.json", "budget.json", "errors.log", "controller_traceback.txt",
                 "metrics.jsonl", "training_log.jsonl", "events.jsonl", "usage.jsonl",
                 "gpu_samples.jsonl", "wandb_link.json", ".stage_complete.json", ".rsi_complete.json"}
        if self.text_enabled:
            names.update({"records.jsonl", "generations.jsonl", "rollout_diagnostics.jsonl", "invalid_group.json"})
        paths = [self.root / name for name in sorted(names) if (self.root / name).is_file()]
        paths += sorted(self.root.glob("environment_*.json"))
        paths += sorted(self.root.glob("logs/*.log"))
        # Freeze the complete expert dataset identity; raw prompts/references are opt-in.
        if (self.root / 'expert_run.json').exists():
            paths += [p for p in (self.root / 'data/manifest.json',) if p.is_file()]
            if self.text_enabled:
                paths += sorted((self.root / 'data').glob('*.jsonl'))
                paths += sorted((self.root / 'data').glob('*/train.jsonl'))
                paths += sorted((self.root / 'data').glob('*/dev.jsonl'))
        if self.text_enabled:
            paths += [self.root / 'review.jsonl'] if (self.root / 'review.jsonl').is_file() else []
        limit = int(os.environ.get("MARGENT_WANDB_ARTIFACT_MAX_BYTES", str(100 * 1024 * 1024)))
        if limit < 1:
            raise ValueError("MARGENT_WANDB_ARTIFACT_MAX_BYTES must be positive")
        export = self.root / "tracking_exports" / self.attempt
        export.mkdir(parents=True, exist_ok=True)
        artifact = self.sdk.Artifact("evidence-" + self.identity["id"], type="experiment-evidence",
                                     metadata={"stage": self.stage, "attempt": self.attempt,
                                               "text_enabled": self.text_enabled, "schema_version": 2})
        index, omitted, used = [], [], 0
        for path in paths:
            name = path.relative_to(self.root).as_posix()
            # Never follow a log symlink into unrelated data.
            if path.is_symlink() or self.root not in path.resolve().parents:
                omitted.append({"path": name, "reason": "outside_root_or_symlink"})
                continue
            if path.stat().st_size + used > limit:
                omitted.append({"path": name, "reason": "artifact_size_limit", "bytes": path.stat().st_size})
                continue
            raw = path.read_bytes()
            text = raw.decode("utf-8", errors="replace")
            parse_error = False
            if path.suffix == ".json":
                try:
                    text = json.dumps(_config(json.loads(text)), ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    # A torn writer is itself useful failure evidence.
                    text, parse_error = redact_text(text), True
            else:
                text = redact_text(text)
            data = text.encode("utf-8")
            target = export / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            artifact.add_file(str(target), name=name)
            used += len(raw)
            index.append({"path": name, "source_sha256": hashlib.sha256(raw).hexdigest(),
                          "export_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                          "json_parse_error": parse_error})
        _write(export / "evidence_index.json", {"files": index, "omitted": omitted, "redacted": True})
        artifact.add_file(str(export / "evidence_index.json"), name="evidence_index.json")
        self.run.log_artifact(artifact)
        self.summary({"evidence/artifact_name": artifact.name, "evidence/files": len(index),
                      "evidence/omitted_files": omitted, "evidence/upload_status": "submitted"})

    def log_text(self, kind, record):
        if self.run is None or not self.text_enabled:
            return
        from .debug_records import (GENERATION_COLUMNS, QUESTION_COLUMNS, ROLLOUT_COLUMNS,
                                    generation_values, question_values, table_row)
        if kind == "generation":
            columns, values = GENERATION_COLUMNS, generation_values(record)
        elif kind == "question":
            columns, values = QUESTION_COLUMNS, question_values(record, record.get("source", "live"))
        elif kind == "rollout":
            columns, values = ROLLOUT_COLUMNS, record
        else:
            raise ValueError(f"Unknown text table: {kind}")
        # SDK incremental tables do not resume their in-memory cursor. Separate
        # attempts keep earlier rows visible instead of replacing them on resume.
        key = f"debug/{kind}s_{self.attempt}"
        count = self.table_counts.get(key, 0)
        if count >= self.table_max_rows:
            self.run.summary[key + "_omitted_rows"] = self.run.summary.get(key + "_omitted_rows", 0) + 1
            if count == self.table_max_rows:
                print(f"[wandb] {key} reached its display row limit; full records remain local", flush=True)
            self.table_counts[key] = count + 1
            return
        if key not in self.tables:
            self.tables[key] = self.sdk.Table(columns=columns, log_mode="INCREMENTAL")
            self.run.summary["debug/table_keys"] = list(self.tables)
        self.tables[key].add_data(*table_row(columns, values, self.table_max_chars))
        self.table_counts[key] = count + 1
        self.pending_tables.add(key)
        self.flush_tables(force=bool(record.get("truncated") or record.get("error")))

    def flush_tables(self, force=False):
        if not self.pending_tables or (not force and self.last_table_flush is not None
                                       and time.monotonic() - self.last_table_flush < 30):
            return
        keys = sorted(self.pending_tables)
        self.run.log({key: self.tables[key] for key in keys})
        self.pending_tables.difference_update(keys)
        self.last_table_flush = time.monotonic()

    def finish(self, status):
        if self.run:
            try:
                self.flush_tables(force=True)
            finally:
                try:
                    self.run.summary.update({"status": status})
                    checkpoints = []
                    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
                        path = self.root / name
                        if path.is_file() and not path.is_symlink():
                            digest = hashlib.sha256()
                            with path.open("rb") as source:
                                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                                    digest.update(chunk)
                            checkpoints.append({"path": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
                    if checkpoints:
                        _write(self.root / "checkpoint_inventory.json", {"files": checkpoints, "weights_uploaded": False})
                        self.summary({"checkpoint/files": checkpoints, "checkpoint/weights_uploaded": False})
                    self.snapshot_artifacts()
                finally:
                    self.run.finish(exit_code=0 if status == "completed" else 1)
