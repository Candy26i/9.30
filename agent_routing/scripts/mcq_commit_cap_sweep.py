#!/usr/bin/env python3
"""The paper's managers trained with another commit cap, next to S_1 (no GPU).

All of them use the round-1 collection of the benchmark and the SFT recipe of S_1; only the commit cap rho differs:
every rescue record is kept and the commit records are cut to int(rho x number of rescue records) ("no cap": all kept).
Reads ``<analysis>/<ckpt>_<bench>_<pool>.json`` (``mcq_verifier_grid_analyze.py``) and writes, per benchmark, one row
per cap: the commit records the cap keeps (computed from the collection counts with the cap rule, not counted from
each manager's label file), the candidate (draft) accuracy, the learned policy's calls and accuracy, and the same
manager's own equal-information references (random questions with the same calls per advisor; the Verifier on its
least confident candidates at the same number of calls, the cut placed inside the pool; always Verifier). Each manager
is compared with its OWN draft and confidences, so rows differ in the draft as well as in the calling rule.

    python scripts/mcq_commit_cap_sweep.py --analysis docs/mcq_rsi_results/rebuttal/verifier_grid/analysis \\
        --out docs/mcq_rsi_results/rebuttal/verifier_grid/commit_cap_sweep.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHES = (("medqa", "MedQA"), ("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("aqua", "AQuA"))
# benchmark -> (cap as printed, cap value or None for "no cap", checkpoint name), in increasing order of the cap
SWEEP = {
    "medqa": (("0", 0.0, "rho0"), ("1", 1.0, "rho1"), ("2", 2.0, "rho2"), ("3", 3.0, "s1"), ("no cap", None, "allc")),
    "mmlu_pro": (("1", 1.0, "rho1"), ("2", 2.0, "s1"), ("no cap", None, "allc")),
    "gpqa": (("1", 1.0, "rho1"), ("2", 2.0, "s1"), ("no cap", None, "allc")),
    "aqua": (("0", 0.0, "rho0"), ("0.5", 0.5, "rho05"), ("1", 1.0, "s1"), ("no cap", None, "allc")),
}
# round-1 collection: (rescue records, commit records); rebuttal/configuration.md, section 1 (GPQA derived there)
RECORDS = {"medqa": (62, 304), "mmlu_pro": (81, 213), "gpqa": (37, 86), "aqua": (132, 217)}


def commit_kept(bench: str, rho: Optional[float]) -> int:
    rescue, commit = RECORDS[bench]
    return commit if rho is None else min(commit, int(rho * rescue))


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _t(test) -> str:
    if test is None:
        return "–"
    p = lambda v: "<.001" if v < 0.001 else f"{v:.3f}".lstrip("0")
    return f"{100 * test['diff']:+.1f} [{100 * test['ci95'][0]:+.1f}, {100 * test['ci95'][1]:+.1f}], {p(test['mcnemar_p'])}"


def rows(analysis: Path, bench: str, pool: str) -> List[Dict[str, Any]]:
    out = []
    for label, rho, ckpt in SWEEP[bench]:
        path = analysis / f"{ckpt}_{bench}_{pool}.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        rule = lambda name, key="accuracy": (d["rules"].get(name) or {}).get(key)
        out.append({"cap": label, "ckpt": ckpt, "n": d["n"], "commit_kept": commit_kept(bench, rho),
                    "commit_all": RECORDS[bench][1], "rescue": RECORDS[bench][0],
                    "candidate": rule("candidate"), "calls": rule("learned", "calls"), "learned": rule("learned"),
                    "random_roles": rule("random_roles"), "confidence_matched": rule("confidence_matched"),
                    "always_verifier": rule("always_verifier"),
                    "learned_vs_gate": d["tests"].get("learned - confidence_matched"),
                    "learned_vs_always": d["tests"].get("learned - always_verifier")})
    return out


def markdown(tables: Dict[str, List[Dict[str, Any]]], pool: str) -> str:
    L = [f"### Commit cap of the manager's training data, pool `{pool}`", "",
         "Managers trained on the same round-1 collection with the S_1 recipe; only the commit cap rho differs (every "
         "rescue record is kept, the commit records are cut to int(rho x rescue records)). The row marked S_1 is the "
         "paper's manager. Accuracy in %, calls per question; every reference in a row is computed from that manager's "
         "own equal-information grid (its own draft and confidences). The confidence gate sends the Verifier to the "
         "manager's least confident candidates, as many as the learned policy calls, with the cut placed inside this "
         "pool; the learned policy's calls can go to another advisor. Differences: pt [95% CI], exact McNemar p.", "",
         "The second column is computed from the round-1 collection counts with the cap rule, not counted from each "
         "manager's label file. Recorded label files or training logs agree with it for S_1 on all four benchmarks, for "
         "MedQA at rho 0 / 1 / 2 and for the no-cap managers of MedQA, MMLU-Pro and GPQA; for MMLU-Pro rho 1, GPQA rho 1 "
         "and AQuA rho 0 / 0.5 / no cap no label file or training log was read (the GPQA collection counts are derived, "
         "see `configuration.md`).", ""]
    for bench, name in BENCHES:
        R = tables.get(bench) or []
        if not R:
            continue
        L += [f"**{name}** (n = {R[0]['n']}; round-1 collection: {R[0]['rescue']} rescue and {R[0]['commit_all']} commit records)", "",
              "| Commit cap | Commit records kept by the cap (computed) | Candidate only | Calls | Learned calling | Random questions, same calls per advisor "
              "| Verifier on its own least confident candidates, same calls | Learned − gate | Always Verifier | Learned − always Verifier |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in R:
            cap = f"{r['cap']} (S_1)" if r["ckpt"] == "s1" else r["cap"]
            L.append(f"| {cap} | {r['commit_kept']} | {_pct(r['candidate'])} | {r['calls']:.2f} | {_pct(r['learned'])} | "
                     f"{_pct(r['random_roles'])} | {_pct(r['confidence_matched'])} | {_t(r['learned_vs_gate'])} | "
                     f"{_pct(r['always_verifier'])} | {_t(r['learned_vs_always'])} |")
        L.append("")
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", default=None, help="markdown file (a .json with the same stem is written next to it)")
    args = ap.parse_args(argv)
    tables = {bench: rows(Path(args.analysis), bench, args.pool) for bench, _ in BENCHES}
    text = markdown(tables, args.pool)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(tables, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
