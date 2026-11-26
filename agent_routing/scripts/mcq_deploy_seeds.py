#!/usr/bin/env python3
"""Paper-protocol (deployed) evaluations of the manager's training-seed replicates (no GPU).

Reads ``<grid-root>/<ckpt>/<bench>/deploy_test/mcq_rsi_eval.json`` (``python -m src.manager.mcq_rsi evaluate``:
free routing, more than one call allowed, the protocol of the main table) for each replicate, takes the original
S_1's deployed run on the same pools from ``paper_assets/data.json`` (``locked`` -> ``v3g/S_1``), and prints each
value and the mean and sample standard deviation over the checkpoints present.

    python scripts/mcq_deploy_seeds.py --grid-root outputs/verifier_grid --ckpts seed1,seed2,seed3 \\
        --reference docs/mcq_rsi_results/paper_assets/data.json --out docs/mcq_rsi_results/rebuttal/verifier_grid/deploy_seeds.md
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHES = (("medqa", "MedQA"), ("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("aqua", "AQuA"))
# label, key in mcq_rsi_eval.json metrics, key in data.json locked rows, scale, format
ROWS = (("Accuracy (%)", "accuracy", "acc", 100, "{:.1f}"),
        ("Advisor calls per question", "calls_per_example", "calls", 1, "{:.3f}"),
        ("Draft accuracy (%)", "initial_draft_accuracy", "draft_acc", 100, "{:.1f}"),
        ("Call rate when the draft is wrong", "call_rate_given_draft_wrong", "call_rate_draft_wrong", 1, "{:.3f}"),
        ("Corrected (% of questions)", "correction_rate", "correction", 100, "{:.1f}"),
        ("Corrupted (% of questions)", "corruption_rate", "corruption", 100, "{:.1f}"),
        ("Valid answers (%)", "valid_answer_rate", "valid", 100, "{:.1f}"))


def load(root: Path, ckpts: List[str], reference: Optional[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    ref = json.loads(Path(reference).read_text(encoding="utf-8")) if reference else None
    for bench, _ in BENCHES:
        if ref and "v3g/S_1" in ref["locked"].get(f"{bench}:test", {}):
            row = ref["locked"][f"{bench}:test"]["v3g/S_1"]
            # data.json stores percentages for acc / draft_acc and fractions for the rest
            out.setdefault(bench, {})["s1"] = {m: (row[k] / 100 if k in ("acc", "draft_acc") else row[k])
                                               for _, m, k, _, _ in ROWS if k in row} | {"n": row.get("n")}
        for ckpt in ckpts:
            path = root / ckpt / bench / "deploy_test" / "mcq_rsi_eval.json"
            if path.is_file():
                out.setdefault(bench, {})[ckpt] = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return out


def markdown(results, order: List[str], title: str) -> str:
    lines = [f"### {title}", "",
             "Paper protocol: the manager routes freely and may call more than once. `s1` is the original checkpoint's run on "
             "the same pools (the S_1 row of the RSI table); `seedK` are re-trainings of the same label file with the "
             "recipe printed in the paper-era logs and training seed K. Each cell lists the checkpoints in that order, then "
             "mean ± sample SD.", ""]
    for bench, name in BENCHES:
        if bench not in results:
            continue
        have = [c for c in order if c in results[bench]]
        n = next((results[bench][c].get("n") for c in have if results[bench][c].get("n")), None)
        lines += [f"**{name}**" + (f" (n = {n})" if n else "") + f"; checkpoints: {', '.join(have)}", "",
                  "| | " + " | ".join(have) + " | mean ± SD |", "|:--|" + "--:|" * (len(have) + 1)]
        for label, key, _, scale, fmt in ROWS:
            vals = [results[bench][c].get(key) for c in have]
            if any(v is None for v in vals):
                continue
            vals = [scale * v for v in vals]
            summary = fmt.format(vals[0]) if len(vals) == 1 else f"{fmt.format(statistics.mean(vals))} ± {fmt.format(statistics.stdev(vals))}"
            lines.append(f"| {label} | " + " | ".join(fmt.format(v) for v in vals) + f" | {summary} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--ckpts", default="seed1,seed2,seed3")
    ap.add_argument("--reference", default=None, help="paper_assets/data.json (adds the original S_1 run as `s1`)")
    ap.add_argument("--title", default="Deployed evaluation of the manager across training seeds, locked tests")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    ckpts = [c for c in args.ckpts.split(",") if c]
    results = load(Path(args.grid_root), ckpts, args.reference)
    if not results:
        raise SystemExit("no deployed evaluation found")
    text = markdown(results, (["s1"] if args.reference else []) + ckpts, args.title)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
