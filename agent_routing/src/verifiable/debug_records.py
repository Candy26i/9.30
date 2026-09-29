"""Explicit text-table schemas and model-free review of saved experiment output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


GENERATION_COLUMNS = [
    "source", "attempt", "time", "question_hash", "benchmark_name", "split", "example_id", "question", "context", "ground_truth",
    "phase", "sequence", "role", "advisor", "operation", "prompt", "text",
    "valid", "valid_output", "truncated", "error", "finish_reason", "max_tokens", "requested_max_tokens", "max_context", "prompt_tokens", "completion_tokens",
    "actual_completion_tokens", "seconds", "tokens_per_second", "cache_hit", "clipped_fields",
]
QUESTION_COLUMNS = [
    "source", "question_hash", "benchmark_name", "split", "example_id", "question", "context", "ground_truth", "direct_text",
    "direct_correct", "direct_valid", "direct_truncated", "branch_count", "successful_branches",
    "preferred_sequence", "policy_text", "policy_correct", "policy_valid", "policy_error",
    "policy_calls", "self_continue_text", "self_continue_correct", "actual_completion_tokens",
    "generation_seconds", "clipped_fields",
]
ROLLOUT_COLUMNS = ["source", "step", "sample", "question_hash", "question", "reward", "advantage",
    "valid", "failure_type", "error", "calls", "root_valid", "root_truncated", "root_text",
    "final_text", "decisions", "manager_outputs", "clipped_fields"]


def question_info(record):
    """Old records keep the question in their direct prompt; never read arbitrary metadata."""
    question = record.get("question")
    if question is None:
        prompt = record.get("independent_prompt", record.get("base_messages", []))
        question = next((m.get("content", "") for m in prompt if m.get("role") == "user"), "")
    return {"question_hash": record.get("question_hash"), "question": question,
            "context": record.get("context", ""), "ground_truth": record.get("ground_truth")}


def _text(value):
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def table_row(columns, values, max_chars):
    """Bound UI payloads explicitly; full output remains in local JSONL files."""
    clipped = []
    row = []
    for name in columns[:-1]:
        value = values.get(name)
        if isinstance(value, (list, dict)):
            value = _text(value)
        if isinstance(value, str) and len(value) > max_chars:
            marker = "\n...[display clipped; full text in local JSONL]...\n"
            remaining = max_chars - len(marker)
            head = remaining // 2
            value = value[:head] + marker + value[-(remaining - head):]
            clipped.append(name)
        row.append(value)
    return row + [", ".join(clipped)]


def generation_values(record):
    values = {k: record.get(k) for k in GENERATION_COLUMNS}
    values["prompt"] = _text(record.get("messages"))
    tokens = record.get("actual_completion_tokens", record.get("completion_tokens", 0))
    seconds = record.get("seconds", 0)
    values["actual_completion_tokens"] = tokens
    values["tokens_per_second"] = tokens / seconds if seconds and seconds > 0 else None
    return values


def question_values(record, source="live"):
    values = {k: record.get(k) for k in QUESTION_COLUMNS}
    values.update(question_info(record), source=source)
    branches, costs = record.get("branches", []), record.get("costs", [])
    values.update(branch_count=len(branches), successful_branches=sum(bool(b.get("correct")) for b in branches),
                  actual_completion_tokens=sum(c.get("actual_completion_tokens", c.get("completion_tokens", 0)) for c in costs),
                  generation_seconds=sum(c.get("seconds", 0) for c in costs))
    policy = record.get("policy", {})
    for name in ("text", "correct", "valid", "error", "calls"):
        values["policy_" + name] = policy.get(name)
    return values


def upload_records(run_dir, output):
    """Create a separate review run; never resume or modify the source experiment."""
    from ..utils.io import write_json
    from .telemetry import Monitor
    from .wandb_tracking import tracking_mode, text_tracking_enabled

    if tracking_mode() == "disabled" or not text_tracking_enabled():
        raise ValueError("Set MARGENT_WANDB_MODE=online or offline and MARGENT_WANDB_TEXT=1")
    source, target = Path(run_dir).resolve(), Path(output).resolve()
    if source == target or source in target.parents:
        raise ValueError("Use a separate review directory outside the source stage")
    if any((p / "loop.json").exists() for p in (target, *target.parents)):
        raise ValueError("Review output must be outside existing experiment loops")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Choose a new, empty review output directory")
    files = [source / n for n in ("records.jsonl", "generations.jsonl") if (source / n).is_file()]
    if not files:
        raise ValueError("No records.jsonl or generations.jsonl in the supplied stage directory")
    # Snapshot bytes once so the recorded hashes describe exactly what was read.
    snapshots = {p.name: p.read_bytes() for p in files}
    rows = {name: [json.loads(line) for line in content.splitlines() if line.strip()]
            for name, content in snapshots.items()}
    if not any(rows.values()):
        raise ValueError("The source stage has no saved output to review")
    manifest_path = source / "run.json"
    original = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    target.mkdir(parents=True, exist_ok=True)
    write_json(str(target / "run.json"), {"config": {
        "purpose": "saved_output_review_no_model_or_training", "source_stage": str(source),
        "source_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()},
        "original_config": original.get("config", {}), "original_harness": original.get("harness"),
    }})
    generations = rows.get("generations.jsonl", [])
    journal_questions = {r.get("question_hash") for r in generations}
    with Monitor(target, "text_review") as monitor:
        for record in rows.get("records.jsonl", []):
            monitor.completed_question(record, source="saved_records")
            # Older runs have no generation journal. Preserve stored text/costs,
            # leaving unavailable prompts, phase and sequence unknown.
            if record.get("question_hash") not in journal_questions:
                for cost in record.get("costs", []):
                    if "text" in cost:
                        monitor.log_text("generation", {**cost, **question_info(record), "source": "legacy_costs"})
        for record in generations:
            monitor.log_text("generation", {**record, "source": "saved_generations"})
    result = json.loads((target / "wandb_link.json").read_text())
    result["text_upload_status"] = "failed" if monitor.wandb_failed else "submitted"
    if monitor.wandb_failed:
        raise RuntimeError("W&B text upload failed; source records are intact. Check the review events.jsonl")
    return result
