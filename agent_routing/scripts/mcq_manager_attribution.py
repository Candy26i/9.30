#!/usr/bin/env python3
"""What training changed in the manager: a trained manager's grid against the untrained base manager's grid (no GPU).

Both grids are ``mcq_verifier_grid.py`` runs on the same pool with the same advisors and advisor cache, so an advisor
request with the same inputs returns the same output for both managers. Questions are joined on the question hash.

  draft        accuracy of the candidate each manager writes before any call, and how often the two agree
  evidence     on the questions where both managers wrote the SAME candidate, every revision prompt (question, draft,
               call turn, advisor output) is identical for the two managers: only the weights that read it differ.
               Accuracy after each forced call (Extractor, Reasoner, Verifier with / without the candidate, and the
               empty-output revision turn) for the base and the trained manager, paired
  selection    which questions get the one call. All selections below are scored on the TRAINED manager's own outcomes
               with the Verifier, at the trained policy's number of calls, so that only the selection signal differs:
                 learned            the trained policy's own choice of questions
                 trained confidence the trained manager's least confident candidates
                 base confidence    the questions on which the UNTRAINED manager is least confident (no training needed)
                 random             uniformly random questions (2,000 draws)
               and the area under the ROC curve of each signal for "the candidate is wrong"
  systems      whole pipelines at the trained policy's number of calls: the untrained manager with a gate on its own
               confidence (grammar-constrained decoding, no training) against the trained manager's learned calling

The base manager does not follow the DRAFT / commit / call format on its own; its grid is grammar-constrained (the share
of first turns equal to unconstrained greedy decoding is reported). Its candidate and its confidence are therefore
those of the base model under the same output grammar the trained manager uses.

    python scripts/mcq_manager_attribution.py --grid-root outputs/verifier_grid --trained s1 --base base --pool test \\
        --out docs/mcq_rsi_results/rebuttal/attribution
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import mcq_verifier_grid_analyze as A  # noqa: E402

BENCHES = (("medqa", "MedQA"), ("mmlu_pro", "MMLU-Pro"), ("gpqa", "GPQA"), ("aqua", "AQuA"))
KINDS = (("verifier", "Verifier, with the candidate"), ("verifier_nocand", "Verifier, candidate removed"),
         ("reasoner", "Reasoner"), ("extractor", "Extractor"), ("null", "Empty-output revision turn"))


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> Optional[float]:
    """P(score of a positive > score of a negative), ties counted one half; None when one class is empty."""
    pos = [s for s, y in zip(scores, positives) if y]
    neg = [s for s, y in zip(scores, positives) if not y]
    if not pos or not neg:
        return None
    order = sorted(set(scores))
    rank, seen = {}, 0
    counts = {v: 0 for v in order}
    for s in scores:
        counts[s] += 1
    for v in order:  # mid-rank of each tied group
        rank[v] = seen + (counts[v] + 1) / 2
        seen += counts[v]
    return (sum(rank[s] for s in pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def least_confident(rows, conf: Sequence[float], k: int) -> List[bool]:
    """Mask of the k questions with the lowest ``conf`` (ties by example id)."""
    order = sorted(range(len(rows)), key=lambda i: (conf[i], rows[i]["id"]))
    chosen = set(order[:k])
    return [i in chosen for i in range(len(rows))]


def _rates(rows) -> Dict[str, Any]:
    wrong = [r for r in rows if not r["commit"]]
    right = [r for r in rows if r["commit"]]
    call = lambda rs: A._mean(r["action"] != "commit" for r in rs) if rs else None
    return {"call_rate": call(rows), "call_rate_candidate_wrong": call(wrong), "call_rate_candidate_right": call(right)}


def attribute(trained_records, base_records, *, draws: int = 2000, n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    t_all, b_all = A.table(trained_records), A.table(base_records)
    by_q = {r["qid"]: r for r in b_all}
    if len(by_q) != len(b_all) or len({r["qid"] for r in t_all}) != len(t_all):
        raise ValueError("duplicate question in a grid")
    if {r["qid"] for r in t_all} != set(by_q):
        raise ValueError("the two grids do not hold the same questions")
    t = t_all
    b = [by_q[r["qid"]] for r in t]
    n = len(t)
    cmp = lambda first, second: A.compare(second, first, n_boot, seed)  # first minus second
    out: Dict[str, Any] = {"n": n}
    for name, rows in (("trained", t), ("base", b)):
        if all(r["root_ok"] is not None for r in rows):
            out[f"{name}_root_agreement"] = {"n_equal": sum(r["root_ok"] for r in rows), "n": n}

    # ---- draft
    same = [i for i in range(n) if t[i]["draft"] == b[i]["draft"]]
    out["draft"] = {"trained": A._mean(r["commit"] for r in t), "base": A._mean(r["commit"] for r in b),
                    "same_candidate": len(same) / n, "n_same": len(same),
                    "test": cmp([r["commit"] for r in t], [r["commit"] for r in b])}

    # ---- evidence use, identical revision prompts
    out["evidence"] = {}
    ts, bs = [t[i] for i in same], [b[i] for i in same]
    for kind, _ in KINDS:
        if not same or kind not in t[0] or kind not in b[0]:
            continue
        tv, bv = [r[kind] for r in ts], [r[kind] for r in bs]
        wrong = [i for i, r in enumerate(ts) if not r["commit"]]
        right = [i for i, r in enumerate(ts) if r["commit"]]
        rate = lambda xs, idx, want: (sum(xs[i] == want for i in idx) / len(idx)) if idx else None
        out["evidence"][kind] = {
            "n": len(same), "candidate": A._mean(r["commit"] for r in ts), "trained": A._mean(tv), "base": A._mean(bv),
            "test": cmp(tv, bv), "agree": A._mean(x == y for x, y in zip(tv, bv)),
            "fixed": {"trained": rate(tv, wrong, True), "base": rate(bv, wrong, True), "n": len(wrong)},
            "broken": {"trained": rate(tv, right, False), "base": rate(bv, right, False), "n": len(right)}}

    # ---- selection signal, scored on the trained manager's outcomes with the Verifier
    m = sum(r["action"] != "commit" for r in t)
    sel: Dict[str, Any] = {"calls": m / n, "n_calls": m, "trained_policy": _rates(t), "base_policy": _rates(b)}
    have_conf = all(r["conf"] is not None for r in t) and all(r["conf"] is not None for r in b)
    if "verifier" in t[0]:
        learned = A.gated_outcomes(t, [r["action"] != "commit" for r in t])
        sel["learned"] = A._mean(learned)
        rnd = A.random_matched(t, {"verifier": m}, draws, seed)
        sel["random"] = {"mean": rnd["mean"], "ci95": rnd["ci95"]}
        if have_conf:
            own = A.gated_outcomes(t, least_confident(t, [r["conf"] for r in t], m))
            other = A.gated_outcomes(t, least_confident(t, [r["conf"] for r in b], m))
            sel.update(trained_confidence=A._mean(own), base_confidence=A._mean(other),
                       tests={"learned - base confidence": cmp(learned, other),
                              "learned - trained confidence": cmp(learned, own),
                              "trained confidence - base confidence": cmp(own, other)})
    if have_conf:
        sel["auroc_candidate_wrong"] = {
            "trained_confidence": auroc([1 - r["conf"] for r in t], [not r["commit"] for r in t]),
            "base_confidence_for_trained_candidate": auroc([1 - r["conf"] for r in b], [not r["commit"] for r in t]),
            "base_confidence_for_base_candidate": auroc([1 - r["conf"] for r in b], [not r["commit"] for r in b]),
            "trained_call_decision": auroc([float(r["action"] != "commit") for r in t], [not r["commit"] for r in t])}
    out["selection"] = sel

    # ---- whole systems at the trained policy's number of calls
    sys_: Dict[str, Any] = {"trained_candidate": A._mean(r["commit"] for r in t), "base_candidate": A._mean(r["commit"] for r in b),
                            "trained_learned": {"accuracy": A._mean(A.learned_outcomes(t)), "calls": m / n}}
    if "verifier" in t[0] and "verifier" in b[0]:
        sys_["trained_always_verifier"] = A._mean(r["verifier"] for r in t)
        sys_["base_always_verifier"] = A._mean(r["verifier"] for r in b)
        sys_["tests"] = {"always Verifier: trained - base": cmp([r["verifier"] for r in t], [r["verifier"] for r in b])}
        if have_conf:
            gate = A.gated_outcomes(b, least_confident(b, [r["conf"] for r in b], m))
            sys_["base_confidence_gate"] = {"accuracy": A._mean(gate), "calls": m / n}
            sys_["tests"]["trained learned calling - base with a confidence gate"] = cmp(A.learned_outcomes(t), gate)
            rb = A.random_matched(b, {"verifier": m}, draws, seed)
            sys_["base_random"] = {"mean": rb["mean"], "ci95": rb["ci95"]}
    out["systems"] = sys_
    return out


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _t(test) -> str:
    if test is None:
        return "–"
    p = lambda v: "<.001" if v < 0.001 else f"{v:.3f}".lstrip("0")
    return f"{100 * test['diff']:+.1f} [{100 * test['ci95'][0]:+.1f}, {100 * test['ci95'][1]:+.1f}], {p(test['p'])} / {p(test['mcnemar_p'])}"


def markdown(results: Dict[str, Dict[str, Any]], trained: str, base: str, pool: str) -> str:
    present = [(b, name) for b, name in BENCHES if b in results]
    R = [results[b] for b, _ in present]
    head = " | ".join(f"{name} (n={results[b]['n']})" for b, name in present)
    row = lambda label, cells: f"| {label} | " + " | ".join(cells) + " |"
    sep = "|:--|" + "--:|" * len(present)
    L = [f"### What training changed in the manager: `{trained}` against the untrained `{base}`, pool `{pool}`", "",
         "Same questions, advisors and advisor cache for both managers. Tests are paired: difference in accuracy, pt "
         "[95% CI], bootstrap p / McNemar p.", ""]
    agree = [(name, r["base_root_agreement"]) for (_, name), r in zip(present, R) if "base_root_agreement" in r]
    if agree:
        forced = any(a["n_equal"] < A.ROOT_AGREEMENT_FLOOR * a["n"] for _, a in agree)
        L += ["The untrained manager's grammar-constrained first turn equals its unconstrained greedy turn on "
              + "; ".join(f"{name} {a['n_equal']}/{a['n']}" for name, a in agree) + " questions"
              + (": on its own it does not follow the draft / commit / call format, so its rows describe the base model "
                 "under the same output grammar as the trained manager." if forced else "."), ""]
    L += ["**1. Draft (candidate before any call)**", "", f"| | {head} |", sep,
          row("Untrained manager, accuracy (%)", [_pct(r["draft"]["base"]) for r in R]),
          row("Trained manager, accuracy (%)", [_pct(r["draft"]["trained"]) for r in R]),
          row("Trained − untrained", [_t(r["draft"]["test"]) for r in R]),
          row("Same candidate (%)", [_pct(r["draft"]["same_candidate"]) for r in R]), ""]
    L += ["**2. Use of evidence** (questions with the same candidate: both managers read an identical revision prompt). "
          "Accuracy (%) after the forced call: untrained → trained; then trained − untrained.", "", f"| Forced call | {head} |", sep]
    for kind, label in KINDS:
        if any(kind in r["evidence"] for r in R):
            L.append(row(label, [f"{_pct(r['evidence'][kind]['base'])} → {_pct(r['evidence'][kind]['trained'])}; {_t(r['evidence'][kind]['test'])}"
                                 if kind in r["evidence"] else "–" for r in R]))
    if any("verifier" in r["evidence"] for r in R):
        e = lambda r, k, who: _pct(r["evidence"]["verifier"][k][who]) if "verifier" in r["evidence"] else "–"
        L += [row("Verifier: wrong candidates fixed (%), untrained / trained", [f"{e(r, 'fixed', 'base')} / {e(r, 'fixed', 'trained')}" for r in R]),
              row("Verifier: right candidates broken (%), untrained / trained", [f"{e(r, 'broken', 'base')} / {e(r, 'broken', 'trained')}" for r in R])]
    L += ["", "**3. Selection signal** (the trained manager's own outcomes with the Verifier, at the learned policy's number "
          "of calls; only the choice of questions differs). Accuracy (%).", "", f"| Questions chosen by | {head} |", sep,
          row("Calls per question", [f"{r['selection']['calls']:.2f}" for r in R]),
          row("The learned policy", [_pct(r["selection"].get("learned")) for r in R]),
          row("The trained manager's confidence", [_pct(r["selection"].get("trained_confidence")) for r in R]),
          row("The untrained manager's confidence", [_pct(r["selection"].get("base_confidence")) for r in R]),
          row("Chance (95% of random draws)", [f"{_pct(r['selection']['random']['mean'])} [{_pct(r['selection']['random']['ci95'][0])}, {_pct(r['selection']['random']['ci95'][1])}]"
                                              if "random" in r["selection"] else "–" for r in R])]
    for key in ("learned - base confidence", "learned - trained confidence", "trained confidence - base confidence"):
        L.append(row(key.replace(" - ", " − ").capitalize(), [_t(r["selection"].get("tests", {}).get(key)) for r in R]))
    au = lambda r, k: (lambda v: "–" if v is None else f"{v:.3f}")(r["selection"].get("auroc_candidate_wrong", {}).get(k))
    L += [row("AUROC for a wrong candidate: trained confidence", [au(r, "trained_confidence") for r in R]),
          row("AUROC for a wrong candidate: untrained confidence", [au(r, "base_confidence_for_trained_candidate") for r in R]),
          row("AUROC for a wrong candidate: the learned call decision", [au(r, "trained_call_decision") for r in R]),
          row("Learned call rate: wrong / right candidates (%)", [f"{_pct(r['selection']['trained_policy']['call_rate_candidate_wrong'])} / "
                                                                  f"{_pct(r['selection']['trained_policy']['call_rate_candidate_right'])}" for r in R]), ""]
    L += ["**4. Whole systems.** Accuracy (%) / advisor calls per question.", "", f"| System | {head} |", sep,
          row("Untrained manager, candidate only", [f"{_pct(r['systems']['base_candidate'])} / 0.00" for r in R]),
          row("Untrained manager + Verifier on its least confident candidates", [f"{_pct(r['systems']['base_confidence_gate']['accuracy'])} / {r['systems']['base_confidence_gate']['calls']:.2f}"
                                                                                  if "base_confidence_gate" in r["systems"] else "–" for r in R]),
          row("Trained manager, learned calling (first call)", [f"{_pct(r['systems']['trained_learned']['accuracy'])} / {r['systems']['trained_learned']['calls']:.2f}" for r in R]),
          row("Untrained manager + always Verifier", [f"{_pct(r['systems'].get('base_always_verifier'))} / 1.00" for r in R]),
          row("Trained manager + always Verifier", [f"{_pct(r['systems'].get('trained_always_verifier'))} / 1.00" for r in R]),
          row("Learned calling − untrained manager with a confidence gate", [_t(r["systems"].get("tests", {}).get("trained learned calling - base with a confidence gate")) for r in R]),
          row("Always Verifier: trained − untrained", [_t(r["systems"].get("tests", {}).get("always Verifier: trained - base")) for r in R])]
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-root", required=True, help="directory with <checkpoint>/<benchmark>/<pool>/grid_records.jsonl")
    ap.add_argument("--trained", default="s1")
    ap.add_argument("--base", default="base")
    ap.add_argument("--pool", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    root, out = Path(args.grid_root), Path(args.out)
    results = {}
    for bench, _ in BENCHES:
        a, b = (root / ck / bench / args.pool / "grid_records.jsonl" for ck in (args.trained, args.base))
        if a.is_file() and b.is_file():
            results[bench] = attribute(A.read_records(a), A.read_records(b), draws=args.draws, n_boot=args.n_boot, seed=args.seed)
    if not results:
        raise SystemExit(f"no benchmark has both {args.trained} and {args.base} grids for pool {args.pool} under {root}")
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.trained}_vs_{args.base}_{args.pool}"
    (out / f"{stem}.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    text = markdown(results, args.trained, args.base, args.pool)
    (out / f"{stem}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
