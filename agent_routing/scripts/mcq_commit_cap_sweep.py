#!/usr/bin/env python3
"""The submission's sibling checkpoints of S_1 stored under another commit cap, next to S_1 (no GPU).

They are the ``sft_<rho>`` / ``sft_allcommit`` adapters of the collection run that produced S_1. Cap rule: every rescue
record is kept and the commit records are cut to int(rho x number of rescue records) ("no cap": all kept). What is on
record about each manager's training (label file, training log) is printed under the table title.
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


def _share(x) -> str:
    return "–" if x is None else ("<.001" if x < 0.001 else f"{x:.3f}".lstrip("0"))


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
                    "random_roles": rule("random_roles"), "random_share": rule("random_roles", "share_at_least_learned"),
                    "confidence_matched": rule("confidence_matched"),
                    "always_verifier": rule("always_verifier"),
                    "learned_vs_gate": d["tests"].get("learned - confidence_matched"),
                    "learned_vs_always": d["tests"].get("learned - always_verifier")})
    return out


def markdown(tables: Dict[str, List[Dict[str, Any]]], pool: str) -> str:
    L = [f"### Commit cap of the manager's training data, pool `{pool}`", "",
         "The submission's sibling checkpoints of S_1 stored under another commit cap rho (the `sft_<rho>` / "
         "`sft_allcommit` adapters of the collection run that produced S_1; every rescue record is kept, the commit "
         "records are cut to int(rho x rescue records)). The row marked S_1 is the registered S_1 of the benchmark (the "
         "paper's manager; for MedQA the same-configuration substitute of the paper's unarchived checkpoint, see "
         "`configuration.md`, section 1). Accuracy in %, calls per question; every reference in a row is computed from "
         "that manager's own equal-information grid (its own draft and confidences). Learned calling is the policy's "
         "first call followed by a commit. The confidence gate sends the Verifier to the manager's least confident "
         "candidates (the manager's probability of its candidate), as many as the learned policy calls, with the cut "
         "placed inside this pool; the learned policy's calls can go to another advisor. At rho = 0 the learned policy "
         "calls on every question, so that gate is always Verifier and the two difference columns hold the same test. "
         "Random questions: mean of 2,000 draws with the same calls per advisor, and the share of those draws that are "
         "at least as accurate as learned calling. Differences: pt [95% CI], exact McNemar p.", "",
         "What is on record about the training of these managers (`configuration.md`, section 1). The second column is "
         "computed from the round-1 collection counts with the cap rule, not counted from each manager's label file. "
         "Recorded label files or training logs agree with it for S_1 on all four benchmarks, for MedQA at rho 0 / 1 / 2 "
         "and for the no-cap managers of MedQA, MMLU-Pro and GPQA (10 of the 15); for GPQA the collection counts were "
         "themselves solved from the row counts of those two training logs, so that agreement is not an independent "
         "check. For MMLU-Pro rho 1, GPQA rho 1 and AQuA rho 0 / 0.5 / no cap no label file or training log was read: "
         "their training data and recipe are known from the adapter names only. Training logs that print the S_1 "
         "settings (lr 1e-5, 10 epochs) are cited for 8 of the 15 (MedQA rho 0 / 1 / 3 / no cap, MMLU-Pro rho 2 / no "
         "cap, GPQA rho 2 / no cap).", ""]
    for bench, name in BENCHES:
        R = tables.get(bench) or []
        if not R:
            continue
        L += [f"**{name}** (n = {R[0]['n']}; round-1 collection: {R[0]['rescue']} rescue and {R[0]['commit_all']} commit records)", "",
              "| Commit cap | Commit records kept by the cap (computed) | Candidate only | Calls | Learned calling (first call) "
              "| Random questions, same calls per advisor | Share of random draws ≥ learned "
              "| Verifier on the manager's least confident candidates, same calls | Learned − gate | Always Verifier | Learned − always Verifier |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for r in R:
            cap = f"{r['cap']} (S_1)" if r["ckpt"] == "s1" else r["cap"]
            L.append(f"| {cap} | {r['commit_kept']} | {_pct(r['candidate'])} | {r['calls']:.2f} | {_pct(r['learned'])} | "
                     f"{_pct(r['random_roles'])} | {_share(r['random_share'])} | {_pct(r['confidence_matched'])} | {_t(r['learned_vs_gate'])} | "
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
