#!/usr/bin/env python3
"""The paper's original fixed-role baselines next to the equal-information rules, on the same test questions (no GPU).

Reads ``<grid-root>/<ckpt>/<bench>/forced_<which>_<pool>/mcq_rsi_forced_*.json`` (``mcq_rsi evaluate --forced``: the
manager does not draft first and the Verifier gets no candidate; "all" = Extractor, Reasoner, Verifier in that order)
and the equal-information analysis ``<analysis>/<ckpt>_<bench>_<pool>.json``, and prints accuracy (%) / calls.
When the per-question files of the forced runs (``manager_forced_*.jsonl``) and the grid records are next to them, the
two protocols are also compared question by question (joined on the example id): paired bootstrap and exact McNemar,
per benchmark and pooled; the accuracy of each forced run is then recomputed from its per-question file and must equal
the summary.

    python scripts/mcq_forced_baselines.py --grid-root outputs/verifier_grid --ckpt s1 \\
        --analysis docs/mcq_rsi_results/rebuttal/verifier_grid/analysis --out docs/mcq_rsi_results/rebuttal/verifier_grid/forced_s1_test.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mcq_verifier_grid_analyze as A  # noqa: E402

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


# paired contrasts, first minus second; "forced:<which>" is an original baseline, the other names are grid rules
CONTRASTS = (("Learned calling − original forced Verifier", "learned", "forced:verifier"),
             ("Learned calling − original forced Extractor, Reasoner, Verifier (3 calls)", "learned", "forced:all"),
             ("Always Verifier with the candidate − original forced Verifier", "always_verifier", "forced:verifier"),
             ("Original forced Verifier − always Verifier, candidate removed", "forced:verifier", "always_verifier_nocand"),
             ("Original forced Reasoner − always Reasoner", "forced:reasoner", "always_reasoner"),
             ("Original forced Extractor − always Extractor", "forced:extractor", "always_extractor"))


def forced(root: Path, ckpt: str, bench: str, which: str, pool: str):
    files = sorted((root / ckpt / bench / f"forced_{which}_{pool}").glob("mcq_rsi_forced_*.json"))
    return json.loads(files[0].read_text(encoding="utf-8"))["metrics"] if files else None


def forced_outcomes(root: Path, ckpt: str, bench: str, which: str, pool: str) -> Optional[Dict[int, bool]]:
    """example id -> correct, from the per-question file of a forced run; checked against the run's summary."""
    files = sorted((root / ckpt / bench / f"forced_{which}_{pool}").glob("manager_forced_*.jsonl"))
    if not files:
        return None
    rows = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    out = {int(r["example_id"]): bool(r["correct"]) for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"{files[0]}: duplicate example id")
    m = forced(root, ckpt, bench, which, pool)
    if m is not None and (m["n"] != len(rows) or abs(m["accuracy"] - sum(out.values()) / len(rows)) > 1e-9
                          or abs(m.get("valid_answer_rate", 1.0) - sum(bool(r["valid_answer"]) for r in rows) / len(rows)) > 1e-9):
        raise ValueError(f"{files[0]}: the per-question file does not give the summary's accuracy / valid share / n")
    return out


def paired(root: Path, ckpt: str, bench: str, pool: str, n_boot: int, seed: int) -> Optional[Dict[str, Any]]:
    """The paired contrasts of one benchmark, or None without the grid records."""
    path = root / ckpt / bench / pool / "grid_records.jsonl"
    if not path.is_file():
        return None
    rows = A.table(A.read_records(path))
    by_rule = {"learned": A.learned_outcomes(rows), "always_verifier": [r.get("verifier") for r in rows],
               "always_verifier_nocand": [r.get("verifier_nocand") for r in rows],
               "always_reasoner": [r.get("reasoner") for r in rows], "always_extractor": [r.get("extractor") for r in rows]}
    for which, _ in FORCED:
        got = forced_outcomes(root, ckpt, bench, which, pool)
        if got is not None:
            if set(got) != {r["id"] for r in rows}:
                raise ValueError(f"{bench}/{which}: the forced run and the grid do not hold the same questions")
            by_rule[f"forced:{which}"] = [got[r["id"]] for r in rows]
    vectors: Dict[str, Any] = {}
    tests, same = {}, {}
    for label, first, second in CONTRASTS:
        a, b = by_rule.get(first), by_rule.get(second)
        if a is None or b is None or None in a or None in b:
            continue
        tests[label] = A._test(vectors, label, b, a, n_boot, seed)
        same[label] = A._mean(x == y for x, y in zip(a, b))
    return {"n": len(rows), "tests": tests, "same_outcome": same, "_vectors": vectors}


def _t(test) -> str:
    if test is None:
        return "–"
    p = lambda v: "<.001" if v < 0.001 else f"{v:.3f}".lstrip("0")
    return f"{100 * test['diff']:+.1f} [{100 * test['ci95'][0]:+.1f}, {100 * test['ci95'][1]:+.1f}], {p(test['p'])} / {p(test['mcnemar_p'])}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--ckpt", default="s1")
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
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
            cells.append(f"{100 * m['accuracy']:.1f} / {m['calls_per_example']:.2f}" + (f" ({100 * valid:.1f}% valid)" if valid < 1.0 else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in GRID:
        cells = [f"{100 * grid[b]['rules'][key]['accuracy']:.1f} / {grid[b]['rules'][key]['calls']:.2f}"
                 if b in grid and key in grid[b]["rules"] else "–" for b, _ in BENCHES]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if missing:
        lines += ["", "Not finished: " + ", ".join(missing) + "."]
    pairs = {b: r for b, _ in BENCHES for r in [paired(root, args.ckpt, b, args.pool, args.n_boot, args.seed)] if r and r["tests"]}
    pooled = A.pooled_tests([r.pop("_vectors") for r in pairs.values()], args.n_boot, args.seed) if len(pairs) > 1 else {}
    if pairs:
        cols = [(b, name) for b, name in BENCHES if b in pairs]
        lines += ["", "**Question by question** (the same questions, joined on the example id). First minus second, accuracy in pt "
                  "[95% CI], bootstrap p / exact McNemar p; in parentheses the share of questions (%) with the same outcome "
                  "under the two rules.", "",
                  "| Contrast | " + " | ".join(f"{name} (n={pairs[b]['n']})" for b, name in cols)
                  + (f" | Pooled (n={next(iter(pooled.values()))['n']})" if pooled else "") + " |",
                  "|:--|" + "--:|" * (len(cols) + bool(pooled))]
        for label, _, _ in CONTRASTS:
            if not any(label in pairs[b]["tests"] for b, _ in cols):
                continue
            cells = [f"{_t(pairs[b]['tests'][label])} ({100 * pairs[b]['same_outcome'][label]:.1f})" if label in pairs[b]["tests"] else "–"
                     for b, _ in cols]
            lines.append(f"| {label} | " + " | ".join(cells) + (f" | {_t(pooled.get(label))}" if pooled else "") + " |")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps({"paired": pairs, "pooled": pooled}, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
