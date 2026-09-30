"""CPU-only verification and score export for the AIME2026 held-out matrix.

Checks every completed cell's question IDs, shards, counts, config, checkpoint, data hash, harness and advisor
identity, recomputes its summary, and writes $EVAL_ROOT/scores.csv. Cells that are missing or unfinished are
listed as incomplete; they are never replaced by another checkpoint.

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


def main():
    root, rsi = Path(os.environ["EVAL_ROOT"]), Path(os.environ["RSI_OUTPUT"])
    data = Path(os.environ["LUNA_DATA"]) / "manager" / f"{BENCH}.jsonl"
    cfg = load_config(os.environ["RSI_CONFIG"])
    bundle = load_expert_bundle(cfg["advisor_expert_bundle"], cfg["base_model"], cfg["base_model_revision"])
    advisor = json.loads((rsi / "advisor_identity.json").read_text())
    assert advisor["expert_bundle"] == bundle and advisor["expert_bundle_sha256"] == expert_bundle_sha256(bundle)
    source = load_rows(str(data), required_split="test")
    expected = {identity(r.question) for r in source}
    assert len(source) == len(expected) == EXPECTED_N
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()

    rows, incomplete = [], []
    for arm in ["base", *os.environ.get("RSI_ARMS", "dynamic static success").split()]:
        label = arm if arm == "base" else f"{arm}_final"
        checkpoint = "Qwen/Qwen3.5-9B" if arm == "base" else str(rsi / arm / "round_2/grpo")
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
        assert len(records) == summary["n"] == EXPECTED_N
        assert {x["question_hash"] for x in records} == expected
        assert run["config"] == cfg and run["checkpoint"] == checkpoint_identity(checkpoint)
        assert json.loads((p / "advisor_identity.json").read_text()) == advisor
        assert run["data_sha256"] == data_sha
        assert run["mode"] == "evaluate" and run["limit"] == 0 and run["harness"] == harness_identity()
        validate_resume_records(records, source, "evaluate")
        observed = summarize(records)
        assert all(summary[k] == observed[k] for k in KEYS)
        shards = {x.stem: json.loads(x.read_text()) for x in (p / "questions").glob("*.json")}
        assert set(shards) == expected and all(shards[x["question_hash"]] == x for x in records)
        rows.append({"checkpoint": label, "benchmark": BENCH, **{k: summary[k] for k in KEYS}})

    if rows:
        with (root / "scores.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"verified_cells": len(rows), "incomplete_cells": incomplete, "scores": rows},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
