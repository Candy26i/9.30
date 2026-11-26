#!/usr/bin/env python3
"""Two managers' learned calling on the same questions (no GPU).

Reads the equal-information grids of two manager checkpoints (``mcq_verifier_grid.py``) and compares the policies
each manager learned: accuracy and calls of its first learned call, which advisor it calls, accuracy gained per call,
and the paired difference in accuracy (95% bootstrap interval, exact McNemar p), per benchmark and pooled.

    python scripts/mcq_policy_compare.py --grid-root outputs/verifier_grid --a s1 --b sucd --a-label "S_1" \\
        --b-label "success distillation" --out docs/mcq_rsi_results/rebuttal/verifier_grid/s1_vs_sucd_policy.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcq_verifier_grid_analyze as A  # noqa: E402

BENCHES = (("medqa", "MedQA"), ("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("aqua", "AQuA"))


def _p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")


def side(rows) -> dict:
    learned = A.learned_outcomes(rows)
    n = len(rows)
    calls = sum(r["action"] != "commit" for r in rows) / n
    cand = A._mean(r["commit"] for r in rows)
    return {"learned": learned, "accuracy": A._mean(learned), "calls": calls, "candidate": cand,
            "mix": [sum(r["action"] == k for r in rows) for k in A.ADVISORS],
            "gain_per_call": (A._mean(learned) - cand) / calls if calls else None,
            "always_verifier": A._mean(r["verifier"] for r in rows)}


def compare(root: Path, a: str, b: str, pool: str, n_boot: int = 10_000, seed: int = 0):
    out, pa, pb = {}, [], []
    for bench, _ in BENCHES:
        fa, fb = (root / c / bench / pool / "grid_records.jsonl" for c in (a, b))
        if not (fa.is_file() and fb.is_file()):
            continue
        ra, rb = A.table(A.read_records(fa)), A.table(A.read_records(fb))
        by = {r["qid"]: r for r in rb}
        if set(by) != {r["qid"] for r in ra}:
            raise ValueError(f"{bench}: the two grids do not hold the same questions")
        rb = [by[r["qid"]] for r in ra]
        sa, sb = side(ra), side(rb)
        out[bench] = {"n": len(ra), "a": sa, "b": sb, "test": A.compare(sb["learned"], sa["learned"], n_boot, seed)}
        pa += sa["learned"]
        pb += sb["learned"]
    pooled = {**A.compare(pb, pa, n_boot, seed), "n": len(pa)} if len(out) > 1 else None
    return out, pooled


def markdown(res, pooled, a_label: str, b_label: str, pool: str) -> str:
    present = [(b, name) for b, name in BENCHES if b in res]
    head = " | ".join(f"{name} (n={res[b]['n']})" for b, name in present)
    row = lambda label, f: f"| {label} | " + " | ".join(f(res[b]) for b, _ in present) + " |"
    acc = lambda s: f"{100 * s['accuracy']:.1f} / {s['calls']:.2f}"
    gpc = lambda s: "–" if s["gain_per_call"] is None else f"{100 * s['gain_per_call']:.1f}"
    t = lambda x: f"{100 * x['diff']:+.1f} [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}], {_p(x['mcnemar_p'])}"
    lines = [f"### Learned calling of two managers on the same questions: {a_label} against {b_label}, pool `{pool}`", "",
             "Each manager's first learned call followed by the forced-commit revision (the equal-information grid). "
             "Accuracy (%) / advisor calls per question.", "", f"| | {head} |", "|:--|" + "--:|" * len(present),
             row(f"{a_label}: learned calling", lambda r: acc(r["a"])),
             row(f"{b_label}: learned calling", lambda r: acc(r["b"])),
             row(f"{a_label}: first call to Extractor / Reasoner / Verifier", lambda r: " / ".join(map(str, r["a"]["mix"]))),
             row(f"{b_label}: first call to Extractor / Reasoner / Verifier", lambda r: " / ".join(map(str, r["b"]["mix"]))),
             row(f"{a_label}: draft accuracy (%)", lambda r: f"{100 * r['a']['candidate']:.1f}"),
             row(f"{b_label}: draft accuracy (%)", lambda r: f"{100 * r['b']['candidate']:.1f}"),
             row(f"{a_label}: accuracy gained per call (pt)", lambda r: gpc(r["a"])),
             row(f"{b_label}: accuracy gained per call (pt)", lambda r: gpc(r["b"])),
             row(f"{b_label}: always Verifier with the candidate (%)", lambda r: f"{100 * r['b']['always_verifier']:.1f}"),
             row(f"{a_label} − {b_label}, learned calling: pt [95% CI], exact McNemar p", lambda r: t(r["test"])), ""]
    if pooled:
        lines.append(f"Pooled over the {pooled['n']} questions: {a_label} − {b_label} = {t(pooled)}.")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-label", default=None)
    ap.add_argument("--b-label", default=None)
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res, pooled = compare(Path(args.grid_root), args.a, args.b, args.pool)
    if not res:
        raise SystemExit("no benchmark has both grids")
    text = markdown(res, pooled, args.a_label or args.a, args.b_label or args.b, args.pool)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
