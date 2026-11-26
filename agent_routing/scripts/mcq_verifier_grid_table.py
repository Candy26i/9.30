#!/usr/bin/env python3
"""One table per manager checkpoint from the per-benchmark analyses of the equal-information grid (no GPU).

Reads ``<analysis>/<ckpt>_<bench>_<pool>.json`` (``mcq_verifier_grid_analyze.py``) and writes, for one checkpoint and
pool, the rules as rows and the benchmarks as columns (accuracy % / advisor calls per question), followed by the
paired tests. Benchmarks whose analysis is missing are left out and named.

With ``--grid-root`` (the directory holding ``<ckpt>/<bench>/<pool>/grid_records.jsonl``) the paired tests are also
computed over the questions of all benchmarks together ("Pooled" column).

    python scripts/mcq_verifier_grid_table.py --analysis runs/verifier_grid/analysis --ckpt s1 --pool test \\
        --grid-root runs/verifier_grid
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
    ("random_roles", "Random questions, same calls per advisor"),
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


def pooled(root: Path, ckpt: str, pool: str, benches, n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """Paired tests over the questions of all ``benches`` together, from the raw grid records."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mcq_verifier_grid_analyze as A

    sets = []
    for bench in benches:
        records = A.read_records(root / ckpt / bench / pool / "grid_records.jsonl")
        dev = root / ckpt / bench / "dev" / "grid_records.jsonl"
        res = A.analyze(records, A.read_records(dev) if pool != "dev" and dev.is_file() else None, draws=200,
                        n_boot=200, seed=seed)
        sets.append(res["_vectors"])
    return A.pooled_tests(sets, n_boot, seed)


def markdown(results: Dict[str, Dict[str, Any]], ckpt: str, pool: str, pooled_tests: Optional[Dict[str, Any]] = None) -> str:
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
    share = [(name, results[b]["rules"]) for b, name in present]
    if all("random_roles" in r and "random_verifier" in r for _, r in share):
        lines += ["Share of 2,000 random draws at least as accurate as the learned calling (one-sided): same calls per advisor "
                  + "; ".join(f"{name} {_p(r['random_roles']['share_at_least_learned'])}" for name, r in share)
                  + ". Verifier-only random subsets against the learned selection sent to the Verifier: "
                  + "; ".join(f"{name} {_p(r['random_verifier']['share_at_least_learned_selection_verifier'])}" for name, r in share)
                  + ".", ""]
    fmt = lambda t: "–" if t is None else (f"{100 * t['diff']:+.1f} [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}], "
                                           f"{_p(t['mcnemar_p'])}")
    extra = [f"Pooled (n={next(iter(pooled_tests.values()))['n']})"] if pooled_tests else []
    lines += [f"| Paired difference in accuracy, pt [95% bootstrap CI], exact McNemar p | {' | '.join([name for _, name in present] + extra)} |",
              "|:--|" + "--:|" * (len(present) + len(extra))]
    for key, label in TESTS:
        if not any(key in results[b]["tests"] for b, _ in present):
            continue
        cells = [fmt(results[b]["tests"].get(key)) for b, _ in present]
        if pooled_tests:
            cells.append(fmt(pooled_tests.get(key)))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", default=None, help="markdown file (default: print only)")
    ap.add_argument("--grid-root", default=None, help="raw grid records, for the pooled column")
    args = ap.parse_args(argv)
    results = load(Path(args.analysis), args.ckpt, args.pool)
    if not results:
        raise SystemExit(f"no analysis for {args.ckpt} / {args.pool} in {args.analysis}")
    together = pooled(Path(args.grid_root), args.ckpt, args.pool, list(results)) if args.grid_root and len(results) > 1 else None
    text = markdown(results, args.ckpt, args.pool, together)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
