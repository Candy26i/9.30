#!/usr/bin/env python3
"""Training-seed replicates of the equal-information comparison (no GPU).

Reads ``<analysis>/<ckpt>_<bench>_<pool>.json`` for several checkpoints that differ only in the manager's training
seed (default: ``s1`` = the paper's manager, ``seed1``, ``seed2``) and reports, per benchmark, each seed's value and
the mean and sample standard deviation over the seeds that are present, for the rows the rebuttal relies on, followed
by the verdict of each paired contrast in every seed.

    python scripts/mcq_verifier_grid_seeds.py --analysis runs/verifier_grid/analysis --ckpts s1,seed1,seed2
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHES = (("medqa", "MedQA"), ("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("aqua", "AQuA"))
ROWS = (
    ("candidate", "accuracy", "Candidate only, accuracy (%)"),
    ("learned", "accuracy", "Learned calling (first call), accuracy (%)"),
    ("learned", "calls", "Learned calling, calls per question"),
    ("random_roles", "accuracy", "Random questions, same calls per advisor, accuracy (%)"),
    ("confidence_matched", "accuracy", "Confidence-gated Verifier, same number of calls, accuracy (%)"),
    ("always_verifier", "accuracy", "Always Verifier with the candidate, accuracy (%)"),
    ("always_verifier_nocand", "accuracy", "Always Verifier, candidate removed, accuracy (%)"),
    ("always_null", "accuracy", "Empty-output revision turn, accuracy (%)"),
)
CONTRASTS = (
    ("learned - always_verifier", "Learned − always Verifier (with candidate)"),
    ("learned - confidence_matched", "Learned − confidence gate (same number of calls)"),
    ("candidate value: always_verifier - always_verifier_nocand", "Always Verifier: with − without the candidate"),
    ("evidence value: always_verifier - always_null", "Always Verifier − empty-output revision"),
)


def load(analysis: Path, ckpts: List[str], pool: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for bench, _ in BENCHES:
        for ckpt in ckpts:
            path = analysis / f"{ckpt}_{bench}_{pool}.json"
            if path.is_file():
                out.setdefault(bench, {})[ckpt] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _mean_sd(values: List[float]) -> str:
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}"


def _p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def markdown(results: Dict[str, Dict[str, Dict[str, Any]]], ckpts: List[str], pool: str) -> str:
    lines = [f"### Training-seed replicates of the manager, pool `{pool}`", "",
             f"Checkpoints: {', '.join(f'`{c}`' for c in ckpts)}. `s1` is the paper's checkpoint; `seedK` is the same label file "
             "retrained with the recipe printed in the paper-era training logs (LoRA r16 / alpha 32, lr 1e-5, 10 epochs, batch 1 x 8) "
             "and training seed K. The original run's seed and maximum sequence length were not logged (command defaults 42 and "
             "4096 are used for the re-trainings), and the re-trainings use the fast linear-attention kernels, so the replicates "
             "differ from `s1` in the seed and possibly in those unlogged details. Each cell lists the checkpoints in that order, "
             "then mean ± sample SD over the ones present.", ""]
    for bench, name in BENCHES:
        if bench not in results:
            continue
        have = [c for c in ckpts if c in results[bench]]
        res = results[bench]
        lines += [f"**{name}** (n = {res[have[0]]['n']}; seeds present: {', '.join(have)})", "",
                  "| | " + " | ".join(have) + " | mean ± SD |", "|:--|" + "--:|" * (len(have) + 1)]
        for rule, field, label in ROWS:
            if not all(rule in res[c]["rules"] for c in have):
                continue
            scale = 100 if field == "accuracy" else 1
            vals = [scale * res[c]["rules"][rule][field] for c in have]
            fmt = (lambda v: f"{v:.1f}") if field == "accuracy" else (lambda v: f"{v:.2f}")
            summary = _mean_sd(vals) if field == "accuracy" else (f"{statistics.mean(vals):.2f} ± {statistics.stdev(vals):.2f}" if len(vals) > 1 else fmt(vals[0]))
            lines.append(f"| {label} | " + " | ".join(fmt(v) for v in vals) + f" | {summary} |")
        if all("random_roles" in res[c]["rules"] for c in have):
            lines.append("| Share of random draws (same calls per advisor) ≥ learned calling | "
                         + " | ".join(_p(res[c]["rules"]["random_roles"]["share_at_least_learned"]) for c in have) + " | |")
        for key, label in CONTRASTS:
            if not all(key in res[c]["tests"] for c in have):
                continue
            cells = [f"{100 * res[c]['tests'][key]['diff']:+.1f} ({_p(res[c]['tests'][key]['mcnemar_p'])})" for c in have]
            diffs = [100 * res[c]["tests"][key]["diff"] for c in have]
            lines.append(f"| {label}, pt (exact McNemar p) | " + " | ".join(cells) + f" | {_mean_sd(diffs)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--ckpts", default="s1,seed1,seed2")
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    ckpts = [c for c in args.ckpts.split(",") if c]
    results = load(Path(args.analysis), ckpts, args.pool)
    if not results:
        raise SystemExit(f"no analysis for {ckpts} / {args.pool} in {args.analysis}")
    text = markdown(results, ckpts, args.pool)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
