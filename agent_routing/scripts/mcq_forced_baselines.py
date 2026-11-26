#!/usr/bin/env python3
"""The paper's original fixed-role baselines next to the equal-information rules, on the same test questions (no GPU).

Reads ``<grid-root>/<ckpt>/<bench>/forced_<which>_<pool>/mcq_rsi_forced_*.json`` (``mcq_rsi evaluate --forced``: the
manager does not draft first and the Verifier gets no candidate; "all" = Extractor, Reasoner, Verifier in that order)
and the equal-information analysis ``<analysis>/<ckpt>_<bench>_<pool>.json``, and prints accuracy (%) / calls.

    python scripts/mcq_forced_baselines.py --grid-root outputs/verifier_grid --ckpt s1 \\
        --analysis docs/mcq_rsi_results/rebuttal/verifier_grid/analysis --out docs/mcq_rsi_results/rebuttal/verifier_grid/forced_s1_test.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

BENCHES = (("medqa", "MedQA"), ("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("aqua", "AQuA"))
FORCED = (("verifier", "Original baseline: forced Verifier (no draft, no candidate)"),
          ("reasoner", "Original baseline: forced Reasoner"),
          ("extractor", "Original baseline: forced Extractor"),
          ("all", "Original baseline: forced Extractor, Reasoner, Verifier"))
GRID = (("candidate", "Equal information: candidate only"),
        ("learned", "Equal information: learned calling (first call)"),
        ("always_verifier", "Equal information: always Verifier, with the candidate"),
        ("always_verifier_nocand", "Equal information: always Verifier, candidate removed"),
        ("always_reasoner", "Equal information: always Reasoner"),
        ("always_extractor", "Equal information: always Extractor"))


def forced(root: Path, ckpt: str, bench: str, which: str, pool: str):
    files = sorted((root / ckpt / bench / f"forced_{which}_{pool}").glob("mcq_rsi_forced_*.json"))
    return json.loads(files[0].read_text(encoding="utf-8"))["metrics"] if files else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--ckpt", default="s1")
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root, analysis = Path(args.grid_root), Path(args.analysis)
    grid = {b: json.loads((analysis / f"{args.ckpt}_{b}_{args.pool}.json").read_text(encoding="utf-8"))
            for b, _ in BENCHES if (analysis / f"{args.ckpt}_{b}_{args.pool}.json").is_file()}
    head = " | ".join(f"{name} (n={grid[b]['n']})" if b in grid else name for b, name in BENCHES)
    lines = [f"### Original fixed-role baselines and equal-information rules: manager `{args.ckpt}`, pool `{args.pool}`", "",
             "Accuracy (%) / advisor calls per question; valid-answer rate in parentheses when below 100%. The original "
             "baselines are the submission's protocol (`evaluate --forced`): the manager does not write a draft before the "
             "forced call and the Verifier receives no candidate. The equal-information rows start from the manager's own "
             "draft and are scored by the forced-commit revision.", "", f"| Rule | {head} |", "|:--|" + "--:|" * len(BENCHES)]
    missing = []
    for which, label in FORCED:
        cells = []
        for b, _ in BENCHES:
            m = forced(root, args.ckpt, b, which, args.pool)
            if m is None:
                cells.append("–")
                missing.append(f"{b}/{which}")
                continue
            valid = m.get("valid_answer_rate", 1.0)
            cells.append(f"{100 * m['accuracy']:.1f} / {m['calls_per_example']:.2f}" + (f" ({100 * valid:.0f}% valid)" if valid < 0.9995 else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in GRID:
        cells = [f"{100 * grid[b]['rules'][key]['accuracy']:.1f} / {grid[b]['rules'][key]['calls']:.2f}"
                 if b in grid and key in grid[b]["rules"] else "–" for b, _ in BENCHES]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if missing:
        lines += ["", "Not finished: " + ", ".join(missing) + "."]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
