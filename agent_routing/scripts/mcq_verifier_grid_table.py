#!/usr/bin/env python3
"""One table per manager checkpoint from the per-benchmark analyses of the equal-information grid (no GPU).

Reads ``<analysis>/<ckpt>_<bench>_<pool>.json`` (``mcq_verifier_grid_analyze.py``) and writes, for one checkpoint and
pool, the rules as rows and the benchmarks as columns (accuracy % / advisor calls per question), followed by the
paired tests. Benchmarks whose analysis is missing are left out and named.

    python scripts/mcq_verifier_grid_table.py --analysis runs/verifier_grid/analysis --ckpt s1 --pool test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHES = (("medqa", "MedQA"), ("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("aqua", "AQuA"))
ROWS = (
    ("candidate", "Candidate only (no call)"),
    ("learned", "Learned calling (first call, then commit)"),
    ("learned_selection_verifier", "Learned selection, Verifier only"),
    ("random_verifier", "Random subset, Verifier only (same budget)"),
    ("confidence_dev_matched", "Confidence-gated Verifier (dev threshold, matched call rate)"),
    ("confidence_dev_best", "Confidence-gated Verifier (dev threshold, best dev accuracy)"),
    ("confidence_matched", "Confidence-gated Verifier (same number of calls, in-pool ranking)"),
    ("always_verifier", "Always Verifier, with the candidate"),
    ("always_verifier_nocand", "Always Verifier, candidate removed"),
    ("always_reasoner", "Always Reasoner"),
    ("always_extractor", "Always Extractor"),
    ("always_null", "Revision turn on an empty Verifier output"),
    ("oracle", "Oracle (best of candidate / one call)"),
)
TESTS = (
    ("learned - always_verifier", "Learned − always Verifier (with candidate)"),
    ("learned - learned_selection_verifier", "Learned − learned selection sent to the Verifier"),
    ("learned - confidence_dev_matched", "Learned − confidence gate (dev threshold, matched)"),
    ("learned - confidence_dev_best", "Learned − confidence gate (dev threshold, best)"),
    ("learned - confidence_matched", "Learned − confidence gate (same number of calls)"),
    ("candidate value: always_verifier - always_verifier_nocand", "Always Verifier: with − without the candidate"),
    ("evidence value: always_verifier - always_null", "Always Verifier − empty-output revision"),
    ("revision without evidence: always_null - candidate", "Empty-output revision − candidate"),
)


def load(analysis: Path, ckpt: str, pool: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for bench, _ in BENCHES:
        path = analysis / f"{ckpt}_{bench}_{pool}.json"
        if path.is_file():
            out[bench] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def cell(rule: Optional[Dict[str, Any]], key: str) -> str:
    if rule is None:
        return "–"
    text = f"{100 * rule['accuracy']:.1f} / {rule['calls']:.2f}"
    if key in ("random_verifier", "random_roles") and "ci95" in rule:
        text += f" [{100 * rule['ci95'][0]:.1f}, {100 * rule['ci95'][1]:.1f}]"
    if "n" in rule:
        text += f" (n={rule['n']})"
    return text


def markdown(results: Dict[str, Dict[str, Any]], ckpt: str, pool: str) -> str:
    present = [(b, name) for b, name in BENCHES if b in results]
    missing = [name for b, name in BENCHES if b not in results]
    head = " | ".join(f"{name} (n={results[b]['n']})" for b, name in present)
    lines = [f"### Equal-information comparison: manager `{ckpt}`, pool `{pool}`", "",
             "Accuracy (%) / advisor calls per question. Every rule starts from the same manager-written candidate, uses "
             "the same checkpoint, decoding and forced-commit revision, and makes at most one call."
             + (f" Not finished yet: {', '.join(missing)}." if missing else ""), "",
             f"| Rule | {head} |", "|:--|" + "--:|" * len(present)]
    for key, label in ROWS:
        if any(key in results[b]["rules"] for b, _ in present):
            lines.append(f"| {label} | " + " | ".join(cell(results[b]["rules"].get(key), key) for b, _ in present) + " |")
    lines += ["", "First call of the learned policy: " + "; ".join(
        f"{name} {results[b]['policy_calls']}/{results[b]['n']} questions ("
        + ", ".join(f"{k[0].upper()} {v}" for k, v in results[b]["policy_call_counts"].items()) + ")"
        + (f", commit forced after {results[b]['rules']['learned']['commit_forced']}"
           if "commit_forced" in results[b]["rules"]["learned"] else "") for b, name in present) + ".", ""]
    agreement = [(name, results[b]["root_agreement"]) for b, name in present if results[b].get("root_agreement")]
    if agreement:
        lines += ["First turn equal to unconstrained greedy decoding: "
                  + "; ".join(f"{name} {a['n_equal']}/{a['n']}" for name, a in agreement) + ".", ""]
    lines += [f"| Paired difference in accuracy, pt [95% CI], bootstrap p / McNemar p | {' | '.join(name for _, name in present)} |",
              "|:--|" + "--:|" * len(present)]
    for key, label in TESTS:
        if not any(key in results[b]["tests"] for b, _ in present):
            continue
        cells = []
        for b, _ in present:
            t = results[b]["tests"].get(key)
            cells.append("–" if t is None else f"{100 * t['diff']:+.1f} [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}], "
                                                f"{_p(t['p'])} / {_p(t['mcnemar_p'])}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", default=None, help="markdown file (default: print only)")
    args = ap.parse_args(argv)
    results = load(Path(args.analysis), args.ckpt, args.pool)
    if not results:
        raise SystemExit(f"no analysis for {args.ckpt} / {args.pool} in {args.analysis}")
    text = markdown(results, args.ckpt, args.pool)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
