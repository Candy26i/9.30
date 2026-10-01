"""M0 feasibility probe on the exact RSI train subset, before any RSI launch.

initial_collection's commit count equals M0's greedy direct-correct count on these questions (same root prompt,
seed rule and temperature 0), so an assess run on the train subset predicts half of the initial gate, measures
truncation and output length, and gives real decode throughput. Only train/dev questions are used; never run this
on a held-out test set.

Usage (from agent_routing/, environment sourced, advisor serving):
    python scripts/probe_m0_subset.py prepare --subset "$RSI_SUBSET" --out "$PROBE"
    CUDA_VISIBLE_DEVICES="$RSI_MANAGER_GPU" python -m src.verifiable.rsi stage assess --config "$RSI_CONFIG" \
        --checkpoint Qwen/Qwen3.5-9B --data "$PROBE/train_as_dev.jsonl" --out "$PROBE/train"
    python scripts/probe_m0_subset.py summarize --subset "$RSI_SUBSET" --out "$PROBE" --config "$RSI_CONFIG"
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.io import write_jsonl  # noqa: E402
from src.verifiable.data import identity, load_rows  # noqa: E402


def prepare(args):
    out = Path(args.out)
    target = out / "train_as_dev.jsonl"
    if target.exists():
        raise SystemExit(f"Probe input exists: {target}")
    rows = load_rows(str(Path(args.subset) / "train.jsonl"), required_split="train")
    for row in rows:
        row.split = "dev"  # assess accepts only split=dev; question content is unchanged
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(target), [r.to_dict() for r in rows])
    print(json.dumps({"probe_input": str(target), "n": len(rows)}))


def summarize(args):
    out, cfg = Path(args.out) / "train", json.loads(Path(args.config).read_text())
    records = {r["question_hash"]: r for r in (json.loads(x) for x in (out / "records.jsonl").read_text().splitlines() if x.strip())}
    summary = json.loads((out / "summary.json").read_text())
    gens = [json.loads(x) for x in (out / "generations.jsonl").read_text().splitlines() if x.strip()]
    live = [g for g in gens if g.get("seconds") and not g.get("cache_hit")]
    answers = [g for g in live if g["role"] == "manager" and g.get("operation") == "answer"]
    roots = [g for g in answers if g.get("phase") == "independent"]
    decisions = [g for g in live if g["role"] == "manager" and g.get("operation") == "decision"]
    advice = [g for g in live if g["role"] == "advisor"]
    tokens = lambda xs: sum(g.get("completion_tokens", 0) for g in xs)
    seconds = lambda xs: sum(g["seconds"] for g in xs)
    mean = lambda xs: round(tokens(xs) / len(xs)) if xs else None
    # GRPO trains on rows sorted by question identity, visited in a seed-shuffled order (rsi_grpo.train_grpo).
    rows = sorted(load_rows(str(Path(args.subset) / "train.jsonl"), required_split="train"), key=lambda r: identity(r.question))
    order = list(range(len(rows)))
    random.Random(cfg["seed"]).shuffle(order)
    grpo = [identity(rows[order[step % len(order)]].question) for step in range(cfg["rl_max_steps"])]
    grpo_unique = sorted(set(grpo))
    commits = summary["independent_correct_n"]
    result = {
        "n": summary["n"],
        "independent_correct_n": commits,
        "policy_correct_n": summary.get("policy_correct_n"),
        "policy_rescued_n": summary.get("policy_rescued_n"),
        "direct_truncated_rate": summary.get("direct_truncated_rate"),
        "direct_valid_rate": summary.get("direct_valid_rate"),
        "mean_calls": summary.get("mean_calls"),
        "root_answer_mean_tokens": mean(roots),
        "root_answers_at_cap": sum(g.get("completion_tokens", 0) >= cfg["max_new_tokens"] for g in roots),
        "revision_mean_tokens": mean([g for g in answers if g.get("phase") != "independent"]),
        "decision_mean_tokens": mean(decisions),
        "advisor_mean_tokens": mean(advice),
        "answer_tokens_per_second": round(tokens(answers) / seconds(answers), 1) if answers else None,
        "generation_seconds_per_question": round(seconds(live) / summary["n"]),
        "grpo_steps": len(grpo),
        "grpo_unique_questions": len(grpo_unique),
        "grpo_unique_questions_any_correct": sum(bool(records[h]["direct_correct"] or records[h]["policy"]["correct"])
                                                 for h in grpo_unique),
        "initial_gate_commits_needed": cfg["pilot_min_commits"],
        "initial_gate_commit_half_passes": commits >= cfg["pilot_min_commits"],
        "note": "Rescues in initial_collection search all 9 branches; policy_rescued_n is one greedy route only.",
    }
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("prepare", "summarize"))
    parser.add_argument("--subset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.action == "summarize" and not args.config:
        parser.error("summarize requires --config")
    {"prepare": prepare, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()
