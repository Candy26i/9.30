"""Budgeted pilot CLI: collect -> SFT -> on-policy GRPO -> recollect.

Separate entry point deliberately leaves the historical SFT-only CLI untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from .data import identity, load_rows, verify_manifest
from .experiment import sft_rows
from .provenance import harness_identity
from .runner import load_config, run_data, validate_stage_artifacts, verify_advisor, checkpoint_identity
from .telemetry import Monitor, atomic_json
from ..utils.io import read_jsonl, write_jsonl

ARMS = ("dynamic", "static", "success")


def prepare_pilot(data_dir, output, train_n=16, dev_n=16):
    if train_n < 1 or dev_n < 1:
        raise ValueError("Positive subset sizes required")
    source_manifest = verify_manifest(data_dir)
    selected = {}
    for split, size in (("train", train_n), ("dev", dev_n)):
        rows = sorted(load_rows(str(Path(data_dir) / f"{split}.jsonl"), required_split=split),
                      key=lambda r: identity(r.question))
        if len(rows) < size:
            raise ValueError(f"Insufficient {split} rows")
        selected[split] = rows[:size]
    if {identity(r.question) for r in selected["train"]} & {identity(r.question) for r in selected["dev"]}:
        raise ValueError("Train/dev overlap")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"source": source_manifest, "selection": "normalized_question_hash_ascending",
                "counts": {k: len(v) for k, v in selected.items()}, "sha256": {}}
    for split, rows in selected.items():
        content = "".join(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in rows)
        path = root / f"{split}.jsonl"
        if path.exists() and path.read_text() != content:
            raise ValueError("Subset changed; use a new data directory")
        path.write_text(content)
        manifest["sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(root / "pilot_data.json", manifest)
    return manifest


def verify_pilot(data_dir):
    root = Path(data_dir)
    manifest = json.loads((root / "pilot_data.json").read_text())
    for name, digest in manifest["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Pilot data changed")
    train = load_rows(str(root / "train.jsonl"), required_split="train")
    dev = load_rows(str(root / "dev.jsonl"), required_split="dev")
    if {identity(r.question) for r in train} & {identity(r.question) for r in dev}:
        raise ValueError("Train/dev overlap")
    return manifest


def build_plan(config_path, data_dir, output, rounds=2, arms=ARMS):
    if rounds < 2 or not arms or len(set(arms)) != len(arms) or set(arms) - set(ARMS):
        raise ValueError("RSI requires >=2 rounds and unique supported arms")
    root, data = Path(output).resolve(), Path(data_dir).resolve()
    cfg = load_config(config_path)
    steps = []
    def add(stage, checkpoint, input_path, out, **extra):
        command = [sys.executable, "-m", "src.verifiable.rsi", "stage", stage,
                   "--config", str(Path(config_path).resolve()), "--checkpoint", str(checkpoint),
                   "--data", str(input_path), "--out", str(out)]
        for key, value in extra.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        steps.append({"stage": stage, "output": str(out), "command": command})
    base = cfg["base_model"]
    add("assess", base, data / "dev.jsonl", root / "initial_dev")
    # Common initial tree: success and MARGENT selectors see identical evidence.
    shared = root / "initial_collection"
    add("collect", base, data / "train.jsonl", shared)
    checkpoints = {arm: base for arm in arms}
    # Round-major scheduling makes a partial pilot show comparable progress.
    for n in range(1, rounds + 1):
        for arm in arms:
            rd = root / arm / f"round_{n}"
            source = shared
            if n > 1:
                fresh = rd / "collection"
                add("collect", checkpoints[arm], data / "train.jsonl", fresh)
                # Static still collects a shadow tree to measure drift and
                # charge its acquisition budget; only round-1 labels train it.
                source = shared if arm == "static" else fresh
            if arm == "success":
                selection = rd / "selection"
                add("select", checkpoints[arm], source / "records.jsonl", selection, selection="success")
                source = selection
            add("sft", checkpoints[arm], source / "sft.jsonl", rd / "sft")
            checkpoints[arm] = rd / "sft"
            add("assess", checkpoints[arm], data / "dev.jsonl", rd / "sft_dev")
            add("grpo", checkpoints[arm], data / "train.jsonl", rd / "grpo")
            checkpoints[arm] = rd / "grpo"
            add("assess", checkpoints[arm], data / "dev.jsonl", rd / "grpo_dev")
    return steps


def validate_step(step):
    stage = step["stage"]
    out = Path(step["output"])
    if stage == "select":
        if not (out / "sft.jsonl").exists():
            raise ValueError("Selected SFT targets missing")
    else:
        validate_stage_artifacts(out, "rl" if stage == "grpo" else stage)


def execute(step, log_dir, remaining):
    out = Path(step["output"])
    done = out / ".rsi_complete.json"
    if done.exists():
        if json.loads(done.read_text())["command"] != step["command"]:
            raise ValueError("Completed stage changed")
        validate_step(step)
        return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logfile = Path(log_dir) / ("_".join(out.parts[-3:]) + ".log")
    print(shlex.join(step["command"]), flush=True)
    start = time.monotonic()
    with logfile.open("a") as log:
        process = subprocess.Popen(step["command"], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            # Poll in short intervals so the controller remains interruptible.
            while process.poll() is None:
                if time.monotonic() - start >= remaining:
                    raise TimeoutError("Pilot wall-time budget exhausted; completed artifacts retained")
                time.sleep(1)
            if process.returncode:
                raise RuntimeError(f"Stage failed ({process.returncode}); inspect {logfile}")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    validate_step(step)
    atomic_json(done, {"command": step["command"], "wall_seconds": time.monotonic() - start})


def run(config_path, data_dir, output, rounds=2, hours=24., arms=ARMS, dry_run=False):
    if not 0 < hours <= 24:
        raise ValueError("Pilot wall-time cap must be in (0, 24] hours")
    cfg = load_config(config_path)
    from .rsi_grpo import validate_rl_config
    validate_rl_config(cfg)
    manifest = verify_pilot(data_dir)
    plan = build_plan(config_path, data_dir, output, rounds, arms)
    if dry_run:
        for step in plan:
            print(shlex.join(step["command"]))
        return plan
    root = Path(output).resolve()
    signature = {"config": cfg, "data": manifest, "harness": harness_identity(),
                 "arms": list(arms), "rounds": rounds, "hours": hours, "plan": plan}
    root.mkdir(parents=True, exist_ok=True)
    runfile = root / "rsi_run.json"
    if runfile.exists():
        if json.loads(runfile.read_text()) != signature:
            raise ValueError("Pilot settings changed; choose a new output directory")
    elif any(root.iterdir()):
        raise ValueError("Nonempty pilot directory without matching manifest")
    else:
        atomic_json(runfile, signature)
    # Persist a wall deadline. Restarting cannot accidentally buy another 24h.
    budgetfile = root / "budget.json"
    if not budgetfile.exists():
        atomic_json(budgetfile, {"deadline_unix": time.time() + hours * 3600})
    deadline = json.loads(budgetfile.read_text())["deadline_unix"]
    if all((Path(step["output"]) / ".rsi_complete.json").exists() for step in plan):
        for step in plan:
            validate_step(step)
        return report(root)
    with Monitor(root, "rsi_controller") as monitor:
        try:
            return _run_plan(cfg, root, plan, deadline, monitor)
        except BaseException as exc:
            state = monitor.state.get("progress", {})
            monitor.summary({"controller_status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                             "failed_stage": state.get("current_stage", "advisor_preflight"),
                             "pilot_complete": False, "error_type": type(exc).__name__, "error": str(exc)})
            raise


def _run_plan(cfg, root, plan, deadline, monitor):
    verify_advisor(cfg, root)
    for index, step in enumerate(plan):
        current_stage = str(Path(step["output"]).relative_to(root))
        monitor.summary({"controller_status": "running", "current_stage": current_stage,
                         "completed_stages": index, "planned_stages": len(plan)})
        monitor.update(stage_index=index + 1, total_stages=len(plan), current_stage=current_stage)
        if time.time() >= deadline:
            report(root)
            raise TimeoutError("Pilot deadline reached; do not treat an incomplete arm as a final result")
        execute(step, root / "logs", deadline - time.time())
        report(root)
        if Path(step["output"]).name == "initial_collection":
            records = read_jsonl(str(Path(step["output"]) / "records.jsonl"))
            rescues = sum(bool(r["preferred_sequence"]) for r in records)
            commits = sum(r["preferred_sequence"] == [] for r in records)
            gate = {"rescues": rescues, "commits": commits,
                    "pass": rescues >= cfg.get("pilot_min_rescues", 0)
                    and commits >= cfg.get("pilot_min_commits", 0)}
            atomic_json(root / "initial_gate.json", gate)
            if not gate["pass"]:
                raise RuntimeError("Insufficient rescue/commit examples for an informative pilot; inspect initial_gate.json")
        if step["stage"] == "grpo":
            from .rsi_grpo import committed_step_directories
            reports = [json.loads((p / "step.json").read_text())
                       for p in committed_step_directories(step["output"])]
            mixed = sum(r["mixed_reward_group"] for r in reports)
            if mixed < cfg.get("pilot_min_mixed_groups", 0):
                raise RuntimeError("No informative outcome learning in this GRPO stage; inspect rewards before spending more pilot budget")
    result = report(root)
    monitor.summary({"controller_status": "completed", "current_stage": "completed",
                     "pilot_complete": result["complete"], "completed_stages": result["completed_stages"],
                     "planned_stages": result["planned_stages"]})
    return result


def report(output):
    root = Path(output)
    run_info = json.loads((root / "rsi_run.json").read_text())
    initial_path = root / "initial_dev/records.jsonl"
    initial = {r["question_hash"]: r for r in read_jsonl(str(initial_path))} if initial_path.exists() else {}
    timeline = []
    for step in run_info["plan"]:
        if step["stage"] != "assess" or not (Path(step["output"]) / ".rsi_complete.json").exists():
            continue
        rows = read_jsonl(str(Path(step["output"]) / "records.jsonl"))
        if set(initial) != {r["question_hash"] for r in rows} or len(rows) != len(initial):
            raise ValueError("Evaluation question sets differ")
        summary = json.loads((Path(step["output"]) / "summary.json").read_text())
        summary.update(stage=str(Path(step["output"]).relative_to(root)),
            delegation_rescues=sum(not r["direct_correct"] and r["policy"]["correct"] for r in rows),
            delegation_corruptions=sum(r["direct_correct"] and not r["policy"]["correct"] for r in rows),
            direct_truncated_n=sum(r.get("direct_truncated", False) for r in rows),
            truncated_manager_generations=sum(c.get("truncated", False) for r in rows for c in r.get("costs", [])
                                              if c.get("role") == "manager"),
            independent_new=sum(not initial[r["question_hash"]]["direct_correct"] and r["direct_correct"] for r in rows),
            independent_regressed=sum(initial[r["question_hash"]]["direct_correct"] and not r["direct_correct"] for r in rows))
        timeline.append(summary)
    rl_steps = []
    from .rsi_grpo import committed_step_directories
    for stage_root in root.glob("*/round_*/grpo"):
        for directory in committed_step_directories(stage_root):
            p = directory / "step.json"
            rl_steps.append({"path": str(p.relative_to(root)), **json.loads(p.read_text())})
    collections = []
    first_path = root / "initial_collection/records.jsonl"
    first = {r["question_hash"]: r for r in read_jsonl(str(first_path))} if first_path.exists() else {}
    for step in run_info["plan"]:
        out = Path(step["output"])
        if step["stage"] != "collect" or not (out / ".rsi_complete.json").exists():
            continue
        rows = read_jsonl(str(out / "records.jsonl"))
        if {r["question_hash"] for r in rows} != set(first):
            raise ValueError("Collection question sets changed")
        old_rescues = {k for k, r in first.items() if r["preferred_sequence"]}
        collections.append({"stage": str(out.relative_to(root)), "scope": "training set mechanism diagnostic",
            "n": len(rows), "rescue_n": sum(bool(r["preferred_sequence"]) for r in rows),
            "commit_n": sum(r["preferred_sequence"] == [] for r in rows),
            "initial_rescued_n": len(old_rescues),
            "initial_rescued_now_direct_n": sum(r["question_hash"] in old_rescues and r["direct_correct"] for r in rows),
            "preferred_label_changed_n": sum(first[r["question_hash"]]["preferred_sequence"] != r["preferred_sequence"] for r in rows),
            "usage": json.loads((out / "summary.json").read_text())["usage"]})
    done = sum((Path(p["output"]) / ".rsi_complete.json").exists() for p in run_info["plan"])
    result = {"scope": "feasibility pilot, not evidence of general RSI or statistical significance",
              "completed_stages": done, "planned_stages": len(run_info["plan"]),
              "complete": done == len(run_info["plan"]), "timeline": timeline,
              "collection_drift": collections,
              "rl_groups": len(rl_steps), "mixed_reward_groups": sum(s["mixed_reward_group"] for s in rl_steps),
              "invalid_rl_rollouts": sum(sum(not v for v in s["protocol_valid"]) for s in rl_steps),
              "test_sets_used": False}
    from .reporting import read_optional, stage_cost
    result["stage_costs"] = []
    for step in run_info["plan"]:
        path = Path(step["output"])
        marker = path / ".rsi_complete.json"
        completed = json.loads(marker.read_text()) if marker.exists() else None
        try:
            events = read_optional(path / "events.jsonl")
            ledger = read_optional(path / "usage.jsonl")
            cost = stage_cost(path)
        except ValueError as exc:
            parsing_error = exc.__cause__ if isinstance(exc.__cause__, json.JSONDecodeError) else exc
            if not isinstance(parsing_error, (json.JSONDecodeError, UnicodeDecodeError)):
                raise
            # A killed writer can leave a torn ledger. Report the uncertainty
            # without modifying its evidence or claiming free computation.
            events, ledger = [], []
            cost = {"stage_path": str(path), "observed_wall_seconds": None,
                    "attempts": None, "accounting_complete": False, "usage_incomplete": True,
                    "accounting_error": type(parsing_error).__name__,
                    "actual_generation_tokens": None, "sft_processed_tokens": None,
                    "sft_supervised_tokens": None,
                    **{f"{role}_actual_{kind}_tokens": None for role in ("manager", "advisor")
                       for kind in ("prompt", "completion")}}
        # Include failed/interrupted attempts and unfinished stages. A missing
        # ledger or closing event means unknown consumption, not precise zero.
        if not any(e.get("event") in {"completed", "failed", "interrupted"}
                   and "wall_seconds" in e for e in events):
            cost["observed_wall_seconds"] = None
        if not ledger:
            for key in cost:
                if key.endswith("_tokens"):
                    cost[key] = None
        last_successful = completed.get("wall_seconds") if completed is not None else None
        cost.update(stage=str(path.relative_to(root)), stage_complete=completed is not None,
                    accounting_complete=cost["accounting_complete"] and completed is not None,
                    observed_activity=bool(events or ledger or completed or
                        any((path / name).exists() and (path / name).stat().st_size
                            for name in ("events.jsonl", "usage.jsonl"))),
                    last_successful_attempt_wall_seconds=last_successful,
                    # Compatibility alias; never sum this with observed time.
                    wall_seconds=last_successful,
                    wall_seconds_scope="compatibility alias for last_successful_attempt_wall_seconds; not cumulative stage cost",
                    observed_wall_seconds_scope="sum of closed Monitor attempts, including failed/interrupted attempts; excludes open attempts and subprocess startup outside Monitor",
                    cost_scope="all observed attempt logs; incomplete accounting is a lower bound, missing evidence is null; time fields are alternative views, not additive")
        result["stage_costs"].append(cost)
    result["paired_final_differences"] = []
    import random
    for control in ("static", "success"):
        paths = [root / arm / f"round_{run_info['rounds']}" / "grpo_dev" for arm in ("dynamic", control)]
        if not all((p / ".rsi_complete.json").exists() for p in paths):
            continue
        a, b = [{r["question_hash"]: r for r in read_jsonl(str(p / "records.jsonl"))} for p in paths]
        if set(a) != set(b) or not a:
            raise ValueError("Paired arms have different evaluation questions")
        for metric, fn in (("independent_accuracy", lambda r: int(r["direct_correct"])),
                           ("policy_accuracy", lambda r: int(r["policy"]["correct"])),
                           ("mean_calls", lambda r: r["policy"]["calls"])):
            diffs = [fn(a[k]) - fn(b[k]) for k in sorted(a)]
            rng = random.Random(1234)
            boot = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(2000))
            result["paired_final_differences"].append({"comparison": f"dynamic-minus-{control}",
                "metric": metric, "difference": sum(diffs) / len(diffs), "n": len(diffs),
                "exploratory_paired_bootstrap_95": [boot[49], boot[1949]],
                "scope": "within one seed; small dev set; not confirmatory or multiple-comparison adjusted"})
    atomic_json(root / "pilot_report.json", result)
    import csv
    with (root / "pilot_timeline.csv").open("w", newline="") as stream:
        fields = ["stage", "n", "independent_accuracy", "policy_accuracy", "mean_calls",
                  "currently_independent_call_rate", "delegation_rescues", "delegation_corruptions",
                  "independent_new", "independent_regressed", "direct_truncated_n"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(timeline)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    subset = commands.add_parser("prepare")
    subset.add_argument("--data-dir", required=True)
    subset.add_argument("--out", required=True)
    subset.add_argument("--train-n", type=int, default=16)
    subset.add_argument("--dev-n", type=int, default=16)
    loop = commands.add_parser("run")
    loop.add_argument("--config", required=True)
    loop.add_argument("--data-dir", required=True)
    loop.add_argument("--out", required=True)
    loop.add_argument("--rounds", type=int, default=2)
    loop.add_argument("--hours", type=float, default=24.)
    loop.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    loop.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("report")
    status.add_argument("--out", required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("stage", choices=("collect", "select", "sft", "grpo", "assess"))
    for name in ("config", "checkpoint", "data", "out"):
        stage.add_argument("--" + name, required=True)
    stage.add_argument("--selection", choices=("counterfactual", "success"), default="counterfactual")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_pilot(args.data_dir, args.out, args.train_n, args.dev_n)
    elif args.command == "run":
        result = run(args.config, args.data_dir, args.out, args.rounds, args.hours, args.arms, args.dry_run)
    elif args.command == "report":
        result = report(args.out)
    else:
        cfg = load_config(args.config)
        if args.stage in {"collect", "assess"}:
            result = run_data(cfg, args.data, args.checkpoint, args.out, args.stage, resume=True)
        elif args.stage == "sft":
            from .training import train_sft
            result = train_sft(cfg, args.checkpoint, args.data, args.out)
        elif args.stage == "grpo":
            from .rsi_grpo import train_grpo
            result = train_grpo(cfg, args.checkpoint, args.data, args.out)
        else:
            records = read_jsonl(args.data)
            rows = sft_rows(records, cfg["seed"], cfg.get("commit_rescue_ratio", 1),
                            cfg.get("distill_solutions", True), args.selection)
            if not rows:
                raise ValueError("No successful SFT targets")
            Path(args.out).mkdir(parents=True, exist_ok=True)
            write_jsonl(str(Path(args.out) / "sft.jsonl"), rows)
            result = {"sft_turns": len(rows), "selection": args.selection}
            atomic_json(Path(args.out) / "selection.json", result)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
