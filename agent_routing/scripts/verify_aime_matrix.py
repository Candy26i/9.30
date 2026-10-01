"""CPU-only verification and score export for the AIME2026 held-out matrix.

Checks every completed cell's question IDs, shards, counts, config, checkpoint, data hash, harness and advisor
identity, recomputes its summary, and writes $EVAL_ROOT/scores.csv. Cells that are missing or unfinished, and arms
whose RSI did not reach round_2/grpo_dev, are listed as incomplete and never replaced by another checkpoint. Exits 1
when any cell is incomplete. Uses explicit checks, so python -O / PYTHONOPTIMIZE cannot disable them.

Usage (from agent_routing/, after sourcing the environment file):
    python scripts/verify_aime_matrix.py
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verifiable.data import identity, load_rows  # noqa: E402
from src.verifiable.experiment import summary as summarize  # noqa: E402
from src.verifiable.provenance import harness_identity  # noqa: E402
from src.verifiable.runner import checkpoint_identity, load_config, validate_resume_records  # noqa: E402
from src.verifiable.serve import expert_bundle_sha256, load_expert_bundle  # noqa: E402

BENCH, EXPECTED_N = "aime2026", 30
KEYS = ("n", "independent_correct_n", "policy_correct_n", "independent_accuracy", "policy_accuracy", "mean_calls")


def require(ok, message):
    if not ok:
        raise SystemExit(f"Verification failed: {message}")


def main():
    root, rsi = Path(os.environ["EVAL_ROOT"]), Path(os.environ["RSI_OUTPUT"])
    scores = root / "scores.csv"
    scores.unlink(missing_ok=True)  # never leave an earlier export behind a failed check
    controller = json.loads((rsi / "run_summary.json").read_text())
    data = Path(os.environ["LUNA_DATA"]) / "manager" / f"{BENCH}.jsonl"
    cfg = load_config(os.environ["RSI_CONFIG"])
    bundle = load_expert_bundle(cfg["advisor_expert_bundle"], cfg["base_model"], cfg["base_model_revision"])
    advisor = json.loads((rsi / "advisor_identity.json").read_text())
    require(advisor["expert_bundle"] == bundle and advisor["expert_bundle_sha256"] == expert_bundle_sha256(bundle),
            "RSI advisor identity differs from the configured expert bundle")
    source = load_rows(str(data), required_split="test")
    expected = {identity(r.question) for r in source}
    require(len(source) == len(expected) == EXPECTED_N, "AIME2026 source must have 30 unique questions")
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()

    rows, incomplete = [], []
    for arm in ["base", *(os.environ.get("RSI_ARMS") or "dynamic static success").split()]:
        label = arm if arm == "base" else f"{arm}_final"
        checkpoint = "Qwen/Qwen3.5-9B" if arm == "base" else str(rsi / arm / "round_2/grpo")
        if arm != "base" and not (rsi / arm / "round_2/grpo_dev/.rsi_complete.json").is_file():
            incomplete.append({"checkpoint": label, "status": "rsi_arm_incomplete"})
            continue
        p = root / label / BENCH
        try:
            status = json.loads((p / "status.json").read_text())
        except OSError:
            status = {}
        if status.get("status") != "completed":
            incomplete.append({"checkpoint": label, "status": status.get("status", "missing")})
            continue
        records = [json.loads(x) for x in (p / "records.jsonl").read_text().splitlines() if x.strip()]
        summary = json.loads((p / "summary.json").read_text())
        run = json.loads((p / "run.json").read_text())
        require(len(records) == summary["n"] == EXPECTED_N, f"{label}: record count")
        require({x["question_hash"] for x in records} == expected, f"{label}: question IDs")
        require(run["config"] == cfg, f"{label}: config")
        require(run["checkpoint"] == checkpoint_identity(checkpoint), f"{label}: checkpoint identity")
        require(json.loads((p / "advisor_identity.json").read_text()) == advisor, f"{label}: advisor identity")
        require(run["data_sha256"] == data_sha, f"{label}: data hash")
        require(run["mode"] == "evaluate" and run["limit"] == 0, f"{label}: mode/limit")
        require(run["harness"] == harness_identity(), f"{label}: harness")
        validate_resume_records(records, source, "evaluate")
        observed = summarize(records)
        require(all(summary[k] == observed[k] for k in KEYS), f"{label}: summary differs from records")
        shards = {x.stem: json.loads(x.read_text()) for x in (p / "questions").glob("*.json")}
        require(set(shards) == expected and all(shards[x["question_hash"]] == x for x in records),
                f"{label}: question shards")
        rows.append({"checkpoint": label, "benchmark": BENCH, **{k: summary[k] for k in KEYS}})

    partial = scores.with_suffix(".csv.partial")
    with partial.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "benchmark", *KEYS])
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(scores)
    print(json.dumps({"rsi_controller_status": controller.get("controller_status"),
                      "rsi_failed_stage": controller.get("failed_stage"),
                      "rsi_pilot_complete": controller.get("pilot_complete"),
                      "verified_cells": len(rows), "incomplete_cells": incomplete, "scores": rows},
                     ensure_ascii=False, indent=2))
    raise SystemExit(1 if incomplete else 0)


if __name__ == "__main__":
    main()
