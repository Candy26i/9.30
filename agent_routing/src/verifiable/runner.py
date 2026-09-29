"""Resumable per-example collection and subprocess-isolated loop stages."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

from ..utils.io import append_jsonl, read_jsonl, write_json, write_jsonl
from .backend import HFBackend, HTTPAdvisors
from .data import identity, load_rows, verify_manifest
from .experiment import collect_one, compare, policy_rollout, root_state, sft_rows, summary
from .telemetry import Monitor, atomic_json, completed_question, question_context, progress, metrics
from .protocol import PROTOCOL_VERSION
from .provenance import harness_identity
from .analysis import conditional_metrics

SFT_ARMS = ("dynamic_sft", "success_sft", "static_sft")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a JSON object")
    if cfg.get("decision_constraint", "none") not in {"none", "finite_actions_v1"}:
        raise ValueError("Unknown decision constraint")
    for key in ("base_model", "advisor_url", "max_new_tokens", "advisor_max_tokens", "max_context",
                "max_seq_len", "max_depth", "seed"):
        if key not in cfg:
            raise ValueError(f"Missing config field: {key}")
    if type(cfg["max_depth"]) is not int or cfg["max_depth"] not in (1, 2, 3):
        raise ValueError("max_depth must be 1, 2 or 3")
    if cfg.get("protocol_version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
        raise ValueError("This runner requires protocol_version=2 and fresh run directories")
    for key in ("max_new_tokens", "advisor_max_tokens", "max_context", "max_seq_len"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.get("temperature", 0) != 0:
        raise ValueError("Manager generation requires temperature=0")
    from .sampling import normalize_generation
    normalize_generation(cfg.get("advisor_generation"))
    if type(cfg.get("decision_max_tokens", 128)) is not int or cfg.get("decision_max_tokens", 128) <= 0:
        raise ValueError("decision_max_tokens must be a positive integer")
    for key in ("seed", "generation_seed"):
        if key in cfg and (type(cfg[key]) is not int or not 0 <= cfg[key] <= 2**32 - 1):
            raise ValueError(f"{key} must be an integer in [0, 2**32 - 1]")
    return cfg


def checkpoint_identity(checkpoint):
    path = Path(checkpoint)
    if not path.is_dir():
        return {"model_id": checkpoint}
    # Adapter weights are small enough to hash; full model shards use stat + config.
    return {p.name: (_digest(p) if p.suffix == ".json" or p.name.startswith("adapter_model") else
                     {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns})
            for p in sorted(path.iterdir()) if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}}


def validate_resume_records(records, rows, mode):
    """Validate completion records before trusting their hashes or replacing JSONL.

    Atomic writes protect against torn bytes, not a valid JSON object with missing
    fields. A hash-only shard must never make a question count as completed.
    """
    cohort = {identity(row.question): row for row in rows}
    seen = set()
    for record in records:
        key = record.get("question_hash") if isinstance(record, dict) else None
        if not isinstance(key, str) or key not in cohort or key in seen:
            raise ValueError("Invalid resume record identities")
        seen.add(key)
        row = cohort[key]
        expected = {"example_id": row.example_id, "benchmark_name": row.benchmark_name,
                    "split": row.split, "ground_truth": row.ground_truth, "protocol_version": PROTOCOL_VERSION}
        if any(record.get(name) != value for name, value in expected.items()):
            raise ValueError(f"Resume record metadata differs from frozen data: {key}")
        for name, value in (("question", row.question), ("context", row.context)):
            if name in record and record[name] != value:
                raise ValueError(f"Resume record {name} differs from frozen data: {key}")
        if (type(record.get("direct_correct")) is not bool or type(record.get("direct_valid")) is not bool
                or not isinstance(record.get("direct_text"), str)
                or (record["direct_correct"] and not record["direct_valid"])):
            raise ValueError(f"Incomplete or inconsistent resume direct result: {key}")
        if not isinstance(record.get("costs"), list) or not record["costs"]:
            raise ValueError(f"Missing resume token accounting: {key}")
        for cost in record["costs"]:
            if not isinstance(cost, dict) or any(type(cost.get(name)) is not int or cost[name] < 0
                                                for name in ("prompt_tokens", "completion_tokens")):
                raise ValueError(f"Invalid resume token accounting: {key}")
        outcomes = []
        if mode in {"collect", "diagnose"}:
            if not isinstance(record.get("branches"), list) or not record["branches"] or "preferred_sequence" not in record:
                raise ValueError(f"Incomplete resume collection: {key}")
            outcomes.extend(record["branches"])
        if mode in {"evaluate", "assess", "diagnose"}:
            outcomes.append(record.get("policy"))
        for outcome in outcomes:
            if (not isinstance(outcome, dict) or type(outcome.get("correct")) is not bool
                    or type(outcome.get("valid")) is not bool or not isinstance(outcome.get("text"), str)
                    or (outcome["correct"] and not outcome["valid"])):
                raise ValueError(f"Incomplete or inconsistent resume outcome: {key}")
        if mode in {"evaluate", "assess", "diagnose"}:
            policy = record["policy"]
            if type(policy.get("calls")) is not int or policy["calls"] < 0:
                raise ValueError(f"Invalid resume policy call count: {key}")
    return seen


def run_data(cfg, data, checkpoint, output, mode, resume=False, limit=0,
             selection="counterfactual", backend=None, advisors=None):
    allowed = {"collect": "train", "diagnose": "dev", "assess": "dev", "evaluate": "test"}
    if mode not in allowed:
        raise ValueError(f"Unknown data mode: {mode}")
    rows = load_rows(data, required_split=allowed[mode])
    rows = sorted(rows, key=lambda r: identity(r.question))
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if limit > 0:
        rows = rows[:limit]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    signature = {"config": cfg, "data_sha256": _digest(data), "checkpoint": checkpoint_identity(checkpoint),
                 "mode": mode, "limit": limit, "selection": selection, "protocol_version": PROTOCOL_VERSION, "harness": harness_identity()}
    meta = root / "run.json"
    if meta.exists():
        if not resume or json.loads(meta.read_text()) != signature:
            raise ValueError(f"Existing run differs or --resume missing: {output}")
    else:
        if any(root.iterdir()):
            raise ValueError("Output directory is not empty and has no matching run manifest")
        write_json(str(meta), signature)
    path = root / "records.jsonl"
    shards = root / "questions"
    if shards.exists():
        files = sorted(shards.glob("*.json"))
        records = [json.loads(p.read_text()) for p in files]
        seen = validate_resume_records(records, rows, mode)
        if any(p.stem != record["question_hash"] for p, record in zip(files, records)):
            raise ValueError("Resume shard filename does not match its question hash")
        write_jsonl(str(path), records)  # Recover a torn append from atomic question shards.
    else:
        records = read_jsonl(str(path)) if path.exists() else []
        seen = validate_resume_records(records, rows, mode)
        for record in records:
            atomic_json(shards / (record["question_hash"] + ".json"), record)
    pending = [r for r in rows if identity(r.question) not in seen]
    with Monitor(output, mode):
        if pending:
            backend = backend or HFBackend(cfg["base_model"], checkpoint, cfg["max_context"], revision=cfg.get("base_model_revision"),
                                           decision_constraint=cfg.get("decision_constraint", "none"))
            if advisors is None:
                frozen = verify_advisor(cfg, root)
                advisors = HTTPAdvisors(cfg["advisor_url"], cfg["advisor_max_tokens"], cfg.get("advisor_models"),
                                       generation_options=cfg.get("advisor_generation"))
                if not cfg.get("external_advisor_identity"):
                    advisors.identity = frozen
        started = time.monotonic()
        for row in pending:
            question_context(row)
            progress(completed_examples=len(records), total_examples=len(rows), question_hash=identity(row.question))
            seed = (cfg.get("generation_seed", 1234) + int(identity(row.question)[:8], 16)) % (2 ** 31)
            if mode in {"evaluate", "assess"}:
                direct, history = root_state(row, backend, cfg, seed)
                from .answers import correct
                policy = policy_rollout(row, backend, advisors, cfg, seed, direct, history)
                record = {"question_hash": identity(row.question), "example_id": row.example_id,
                          "benchmark_name": row.benchmark_name, "split": row.split, "protocol_version": PROTOCOL_VERSION,
                          "direct_correct": bool(direct["valid"] and correct(direct["text"], row.ground_truth)),
                          "direct_valid": direct["valid"], "direct_truncated": direct.get("truncated", False),
                          "direct_text": direct["text"], "ground_truth": row.ground_truth, "policy": policy,
                          "costs": [{"role": "manager", **direct}] + policy["costs"]}
            else:
                record = collect_one(row, backend, advisors, cfg, seed, evaluate_policy=mode == "diagnose")
            record.update(question=row.question, context=row.context)
            atomic_json(shards / (record["question_hash"] + ".json"), record)
            append_jsonl(str(path), [record])
            records.append(record)
            completed_question(record)
            progress(completed_examples=len(records), total_examples=len(rows))
            print(f"[{mode}] {len(records)}/{len(rows)} direct={record['direct_correct']}", flush=True)
        result = summary(records)
        result.update(checkpoint=checkpoint, verification_scope="terminal_answer", mode=mode)
        if mode == "collect":
            turns = sft_rows(records, cfg["seed"], cfg.get("commit_rescue_ratio", 1),
                             cfg.get("distill_solutions", True), selection)
            if not turns:
                raise ValueError("No successful trajectories: inspect pilot before launching training")
            write_jsonl(str(root / "sft.jsonl"), turns)
            result["sft_turns"] = len(turns)
        append_jsonl(str(root / "attempts.jsonl"), [{"completed_examples_this_attempt": len(pending),
                                                   "wall_seconds": time.monotonic() - started}])
        write_json(str(root / "summary.json"), result)
        metrics(result, "eval", diagnostic_step=len(records))
        return result


def build_plan(config_path, data_dir, output, arm, rounds, initial=None):
    if arm not in SFT_ARMS:
        raise ValueError("Protocol v2 supports dynamic_sft, success_sft and static_sft; legacy GRPO arms are disabled")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    root = Path(output).resolve()
    data = Path(data_dir).resolve()
    cfg = load_config(config_path)
    checkpoint = initial or cfg["base_model"]
    prefix = [sys.executable, "-m", "src.verifiable"]
    common = ["--config", str(Path(config_path).resolve())]
    plan = []

    def add(stage, input_path, ckpt, out, extra=()):
        command = prefix + [stage] + common + ["--data", str(input_path), "--checkpoint", ckpt,
                  "--out", str(out)] + list(extra)
        plan.append({"stage": stage, "output": str(out), "command": command})

    add("diagnose", data / "dev.jsonl", checkpoint, root / "initial_dev", ["--resume"])
    for n in range(1, rounds + 1):
        rd = root / f"round_{n}"
        collection = root / "round_1" / "collection" if arm == "static_sft" else rd / "collection"
        if n == 1 or arm != "static_sft":
            add("collect", data / "train.jsonl", checkpoint, collection,
                ["--resume", "--selection", "success" if arm == "success_sft" else "counterfactual"])
        add("sft", collection / "sft.jsonl", checkpoint, rd / "sft")
        checkpoint = str(rd / "sft")
        add("diagnose", data / "dev.jsonl", checkpoint, rd / "sft_dev", ["--resume"])
    return plan


def run_loop(config_path, data_dir, output, arm, rounds, initial=None, resume=False, dry_run=False):
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    manifest = verify_manifest(data_dir)
    plan = build_plan(config_path, data_dir, output, arm, rounds, initial)
    if dry_run:
        for step in plan:
            print(shlex.join(step["command"]))
        return plan
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    signature = {"harness": harness_identity(), "config": load_config(config_path), "data_manifest": manifest, "arm": arm,
                 "rounds": rounds, "initial": initial, "plan": plan}
    runfile = root / "loop.json"
    if runfile.exists():
        if not resume or json.loads(runfile.read_text()) != signature:
            raise ValueError("Loop settings changed or --resume is missing; choose another output")
    else:
        if any(root.iterdir()):
            raise ValueError("Loop output is not empty")
        write_json(str(runfile), signature)
    with Monitor(root, "loop"):
        for i, step in enumerate(plan):
            progress(stage_index=i + 1, total_stages=len(plan), current_stage=step["stage"], output=step["output"])
            if not (Path(step["output"]) / ".stage_complete.json").exists():
                verify_advisor(signature["config"], root)
            execute_stage(step, root / "logs")
            if step["stage"] == "diagnose":
                log_diagnostic(root, step, sum(p["stage"] == "diagnose" for p in plan[:i + 1]) - 1)
        checkpoints = [step for step in plan if step["stage"] == "diagnose"]
        reports = []
        before = read_jsonl(str(Path(checkpoints[0]["output"]) / "records.jsonl"))
        for step in checkpoints[1:]:
            after = read_jsonl(str(Path(step["output"]) / "records.jsonl"))
            reports.append({"checkpoint": step["output"], "vs_initial": compare(before, after),
                            "summary": json.loads((Path(step["output"]) / "summary.json").read_text())})
        write_json(str(root / "loop_report.json"), reports)
        return reports


def log_diagnostic(root, step, index):
    """Publish each completed checkpoint immediately, including resumed stages."""
    output = Path(step["output"])
    result = json.loads((output / "summary.json").read_text())
    metrics(result, "eval", diagnostic_step=index)
    initial = Path(root) / "initial_dev" / "records.jsonl"
    before = read_jsonl(str(initial))
    after = read_jsonl(str(output / "records.jsonl"))
    comparison = compare(before, after)
    rescued = sum(not r["direct_correct"] and any(b["correct"] for b in r["branches"]) for r in before)
    gained = len(comparison["previously_rescued_now_independent"])
    values = {"initial_rescued_n": rescued, "rescued_now_independent_n": gained,
              "newly_solved_n": len(comparison["independent"]["newly_solved"]),
              "regressed_n": len(comparison["independent"]["regressed"])}
    if rescued:
        values["rescued_now_independent_rate"] = gained / rescued
    metrics(values, "internalization", diagnostic_step=index)
    metrics(conditional_metrics(after, before), "delegation", diagnostic_step=index)


def execute_stage(step, log_dir):
    """Keep console output and propagate failure; never mark a failed stage done."""
    out = Path(step["output"])
    done = out / ".stage_complete.json"
    if done.exists():
        if json.loads(done.read_text())["command"] != step["command"]:
            raise ValueError("Completed stage command changed")
        validate_stage_artifacts(out, step["stage"])
        return
    print(shlex.join(step["command"]), flush=True)
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    name = "_".join(out.parts[-3:]) + ".log"
    start = time.monotonic()
    with (logs / name).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(step["command"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, step["command"])
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
    validate_stage_artifacts(out, step["stage"])
    atomic_json(done, {"command": step["command"], "wall_seconds": time.monotonic() - start})


def validate_stage_artifacts(output, stage):
    output = Path(output)
    names = ["training_metrics.json", "adapter_config.json"] if stage in {"sft", "rl"} else ["summary.json", "records.jsonl"]
    if stage == "collect":
        names.append("sft.jsonl")
    if stage in {"sft", "rl"} and not any((output / n).exists() for n in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"Adapter weights missing from completed stage: {output}")
    for name in names:
        if not (output / name).exists():
            raise ValueError(f"Completed stage artifact missing: {output / name}")


def evaluate_suite(run_dir, data_dir, dry_run=False):
    root = Path(run_dir).resolve()
    run = json.loads((root / "loop.json").read_text())
    if verify_manifest(data_dir) != run["data_manifest"]:
        raise ValueError("External evaluation data differ from the frozen experiment manifest")
    if run.get("harness") != harness_identity():
        raise ValueError("Harness changed since training; evaluate with the exact recorded code")
    training = [p for p in run["plan"] if p["stage"] in {"sft", "rl"}]
    if not training or not all((Path(p["output"]) / ".stage_complete.json").exists() for p in run["plan"]):
        raise ValueError("Complete the planned loop before the locked initial/final external evaluation")
    config_path = root / "evaluation_config.json"
    plan = []
    for label, checkpoint in [("initial", run["initial"] or run["config"]["base_model"]),
                              ("final", training[-1]["output"])]:
        for name in ("aime2026", "beyondaime"):
            output = root / "test" / label / name
            plan.append({"stage": "evaluate", "output": str(output), "command": [sys.executable, "-m",
                "src.verifiable", "evaluate", "--config", str(config_path), "--data",
                str(Path(data_dir).resolve() / f"{name}.jsonl"), "--checkpoint", checkpoint,
                "--out", str(output), "--resume"]})
    if dry_run:
        for step in plan:
            print(shlex.join(step["command"]))
        return plan
    atomic_json(config_path, run["config"])
    with Monitor(root / "test", "evaluate_suite"):
        for i, step in enumerate(plan):
            progress(stage_index=i + 1, total_stages=len(plan), output=step["output"])
            if not (Path(step["output"]) / ".stage_complete.json").exists():
                verify_advisor(run["config"], root)
            execute_stage(step, root / "logs")
            path = Path(step["output"])
            result = json.loads((path / "summary.json").read_text())
            metrics(result, f"test/{path.parent.name}/{path.name}", diagnostic_step=i)
    return {"evaluations": [p["output"] for p in plan]}


def verify_advisor(config, root):
    import requests
    response = requests.get(config["advisor_url"].rstrip("/") + "/health", timeout=15)
    response.raise_for_status()
    try:
        value = response.json()
    except ValueError:
        value = {}
    identity = value.get("margent_advisor")
    if identity is None:
        # External servers may provide an explicit immutable identity in config.
        identity = config.get("external_advisor_identity")
        if identity is None:
            raise ValueError("Advisor identity unavailable: use the bundled server or declare external_advisor_identity")
    expected_model = config.get("advisor_base_model", config.get("base_model"))
    if identity.get("harness") is not None and identity["harness"] != harness_identity():
        raise ValueError("Subagent server uses a different harness/environment; restart it with the frozen version")
    if expected_model and identity.get("model") and identity["model"] != expected_model:
        raise ValueError("Advisor model differs from configured advisor_base_model/base_model")
    expected_revision = config.get("advisor_revision", config.get("base_model_revision"))
    actual_revision = identity.get("requested_revision") or identity.get("resolved_revision")
    if expected_revision and actual_revision and actual_revision != expected_revision:
        raise ValueError("Advisor revision differs from frozen configuration")
    path = Path(root) / "advisor_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Frozen advisor identity changed across stages; use the original advisor")
    atomic_json(path, identity)
    return identity
