#!/usr/bin/env python3
"""Statistics of an equal-information delegation grid (no GPU).

Input: ``grid_records.jsonl`` of ``mcq_verifier_grid.py``, or any RSI collection file with ``root_mode="policy"``
(``counterfactual_records.jsonl``): both hold, per question, the manager's candidate, the outcome of a forced call to
each advisor from that same candidate (the Verifier with the candidate), the policy's own action and the draft-key
distribution. Every rule below therefore uses the same checkpoint, candidate, decoding and revision prompt; only the
rule that decides which questions get a call differs.

Rules (accuracy and calls per question; every rule makes at most one call per question):

  candidate            never call
  learned              the policy's FIRST action (its advisor, or commit) followed by the forced-commit revision. The
                       deployed policy may call again after an advisor answers; this rule and every other rule here
                       stop after one call, so that all of them are scored by the same revision step. The note reports
                       how often the commit had to be forced.
  learned -> Verifier  the policy's selection of questions, always sent to the Verifier
  random (roles)       the policy's number of calls per advisor, on random questions (2,000 draws: mean, 95% interval,
                       share of draws at least as accurate as the learned rule)
  random (Verifier)    the policy's total number of calls, all to the Verifier, on random questions
  confidence           call the Verifier on the k least confident candidates (confidence = renormalised probability
                       of the candidate key); k = the policy's number of calls, and the whole curve k = 0..n
  confidence (dev)     the same rule with the threshold fixed on a dev grid (``--dev``): the threshold that matches
                       the policy's dev call rate, and the dev-accuracy-maximising one. Thresholds lie between distinct
                       dev confidences. Questions that are also in the dev grid (same question hash) are left out of
                       these two rules and their tests (MMLU-Pro ``test_paper`` contains the whole dev pool)
  always <advisor>     always call that advisor (``verifier_nocand``: the candidate removed from the Verifier's input)
  null                 the Verifier call turn with an empty tool output: no advisor call, one extra revision turn
  oracle               best of {candidate, one call}

Paired tests against the learned rule use the paper's paired bootstrap and exact McNemar (``mcq_rsi_analysis``).

    python scripts/mcq_verifier_grid_analyze.py --records runs/verifier_grid/s1/medqa/test/grid_records.jsonl \\
        --dev runs/verifier_grid/s1/medqa/dev/grid_records.jsonl --out runs/verifier_grid/analysis --label s1_medqa_test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mcq_rsi_analysis import mcnemar, paired_bootstrap, percentile  # noqa: E402

ADVISORS = ("extractor", "reasoner", "verifier")
EXTRA = ("verifier_nocand", "null")


def read_records(path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def confidence(policy_action: Optional[Dict[str, Any]], draft: Optional[str]) -> Optional[float]:
    """Renormalised probability of the candidate key, or None when the key distribution was not recorded."""
    klp = (policy_action or {}).get("key_logprobs")
    if not klp or draft not in klp:
        return None
    top = max(klp.values())
    z = sum(math.exp(v - top) for v in klp.values())
    return math.exp(klp[draft] - top) / z


def table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per question: candidate outcome, the outcome of each single forced call, policy action, confidence."""
    rows = []
    for r in records:
        if r.get("root_mode", "policy") != "policy" or not r.get("policy_action"):
            raise ValueError(f"example {r.get('example_id')}: not a policy-root record")
        if not r.get("direct_valid", True):
            raise ValueError(f"example {r.get('example_id')}: invalid root")
        single = [b for b in r["branches"] if len(b["sequence"]) == 1]
        one = {b["sequence"][0]: bool(b["correct"]) for b in single}
        # would_call: without forcing, the revision after this call would not have been the commit
        again = {b["sequence"][0]: (b.get("revision") or {}).get("would_call") for b in single}
        extra = {b["kind"]: bool(b["correct"]) for b in r.get("extra_branches") or []}
        action = r["policy_action"]["action"]
        check = r.get("root_check")
        if action != "commit" and action not in one:
            raise ValueError(f"example {r['example_id']}: the policy calls {action} but that branch is missing")
        rows.append({"id": int(r["example_id"]), "qid": r.get("question_hash") or int(r["example_id"]),
                     "commit": bool(r["direct_correct"]), **one, **extra, "action": action, "again": again,
                     "root_ok": (check.get("mismatch") == 0) if check else None,
                     "conf": confidence(r["policy_action"], r["direct_pred"])})
    kinds = [k for k in ADVISORS + EXTRA if any(k in row for row in rows)]
    for k in kinds:
        if not all(k in row for row in rows):
            raise ValueError(f"branch {k} is missing for some questions")
    return rows


def _mean(xs) -> float:
    xs = list(xs)
    return sum(map(float, xs)) / len(xs)


def learned_outcomes(rows) -> List[bool]:
    return [row["commit"] if row["action"] == "commit" else row[row["action"]] for row in rows]


def gated_outcomes(rows, called: Sequence[bool], kind: str = "verifier") -> List[bool]:
    return [row[kind] if c else row["commit"] for row, c in zip(rows, called)]


def random_matched(rows, counts: Dict[str, int], draws: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Accuracy when ``counts[advisor]`` calls go to uniformly random distinct questions."""
    n = len(rows)
    total = sum(counts.values())
    if total > n:
        raise ValueError("more calls than questions")
    base = sum(row["commit"] for row in rows)
    delta = {k: [int(row[k]) - int(row["commit"]) for row in rows] for k in counts}
    expected = (base + sum(c / n * sum(delta[k]) for k, c in counts.items())) / n
    rng = random.Random(seed)
    order = [k for k, c in counts.items() for _ in range(c)]
    accs = []
    for _ in range(draws):
        picked = rng.sample(range(n), total)
        accs.append((base + sum(delta[k][i] for k, i in zip(order, picked))) / n)
    accs.sort()
    return {"expected": expected, "mean": _mean(accs), "ci95": [percentile(accs, 0.025), percentile(accs, 0.975)],
            "draws": draws, "seed": seed, "sorted": accs}


def confidence_order(rows) -> Optional[List[int]]:
    """Question indices from the least to the most confident candidate (ties by example id); None without confidences."""
    if any(row["conf"] is None for row in rows):
        return None
    return sorted(range(len(rows)), key=lambda i: (rows[i]["conf"], rows[i]["id"]))


def confidence_curve(rows, kind: str = "verifier") -> Optional[List[float]]:
    """curve[k] = accuracy when the k least confident candidates get a ``kind`` call (k = 0..n)."""
    order = confidence_order(rows)
    if order is None:
        return None
    n = len(rows)
    correct = sum(row["commit"] for row in rows)
    curve = [correct / n]
    for i in order:
        correct += int(rows[i][kind]) - int(rows[i]["commit"])
        curve.append(correct / n)
    return curve


def threshold_for_k(rows, k: int) -> float:
    """A confidence threshold t such that exactly the k least confident candidates satisfy conf < t (up to ties)."""
    order = confidence_order(rows)
    confs = [rows[i]["conf"] for i in order]
    if k <= 0:
        return 0.0
    if k >= len(confs):
        return math.inf
    mid = (confs[k - 1] + confs[k]) / 2
    return mid if mid > confs[k - 1] else confs[k]  # two adjacent floats: the midpoint can round down


def compare(a: Sequence[bool], b: Sequence[bool], n_boot: int, seed: int) -> Dict[str, Any]:
    """b minus a: paired bootstrap of the accuracy difference and exact McNemar."""
    boot = paired_bootstrap([float(x) for x in a], [float(x) for x in b], n_boot=n_boot, seed=seed)
    return {"diff": boot["diff"], "ci95": boot["ci95"], "p": boot["p_two_sided"], "mcnemar_p": mcnemar(a, b)["exact_p"]}


def analyze(records: Sequence[Dict[str, Any]], dev_records: Optional[Sequence[Dict[str, Any]]] = None, *,
            draws: int = 2000, n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    rows = table(records)
    n = len(rows)
    kinds = [k for k in ADVISORS + EXTRA if k in rows[0]]
    learned = learned_outcomes(rows)
    called = [row["action"] != "commit" for row in rows]
    counts = {k: sum(row["action"] == k for row in rows) for k in ADVISORS if k in kinds}
    m = sum(called)
    commit = [row["commit"] for row in rows]
    out: Dict[str, Any] = {"n": n, "policy_calls": m, "policy_call_counts": counts, "rules": {}, "tests": {}}
    rule = lambda name, acc, calls, **kw: out["rules"].__setitem__(name, {"accuracy": acc, "calls": calls, **kw})
    if all(row["root_ok"] is not None for row in rows):
        # the grid's first turn is grammar-constrained; this is how often plain greedy decoding writes the same turn
        out["root_agreement"] = {"n_equal": sum(row["root_ok"] for row in rows), "n": n}

    rule("candidate", _mean(commit), 0.0)
    forced = [row["again"].get(row["action"]) for row in rows if row["action"] != "commit"]
    if forced and all(f is not None for f in forced):
        rule("learned", _mean(learned), m / n, commit_forced=sum(map(bool, forced)), commit_forced_share=_mean(forced))
    else:
        rule("learned", _mean(learned), m / n)
    for k in kinds:
        if k == "null":  # the call turn with an empty tool output: no advisor request is made
            rule("always_null", _mean(row[k] for row in rows), 0.0,
                 note="one extra manager revision turn on an empty Verifier output; no advisor call")
        else:
            rule(f"always_{k}", _mean(row[k] for row in rows), 1.0)
    one_call = [k for k in ADVISORS if k in kinds]
    rule("oracle", _mean(row["commit"] or any(row[k] for k in one_call) for row in rows),
         _mean((not row["commit"]) and any(row[k] for k in one_call) for row in rows))

    if "verifier" in kinds:
        learned_v = gated_outcomes(rows, called)
        rule("learned_selection_verifier", _mean(learned_v), m / n)
        out["tests"]["learned - always_verifier"] = compare([row["verifier"] for row in rows], learned, n_boot, seed)
        out["tests"]["learned - learned_selection_verifier"] = compare(learned_v, learned, n_boot, seed)
        r = random_matched(rows, {"verifier": m}, draws, seed)
        acc_sorted = r.pop("sorted")
        rule("random_verifier", r["mean"], m / n, expected=r["expected"], ci95=r["ci95"], draws=draws,
             share_at_least_learned=sum(a >= _mean(learned) for a in acc_sorted) / draws,
             share_at_least_learned_selection_verifier=sum(a >= _mean(learned_v) for a in acc_sorted) / draws)
        for extra, name in (("verifier_nocand", "candidate value: always_verifier - always_verifier_nocand"),
                            ("null", "evidence value: always_verifier - always_null")):
            if extra in kinds:
                out["tests"][name] = compare([row[extra] for row in rows], [row["verifier"] for row in rows], n_boot, seed)
        if "null" in kinds:
            out["tests"]["revision without evidence: always_null - candidate"] = compare(
                commit, [row["null"] for row in rows], n_boot, seed)
    if counts and m:
        r = random_matched(rows, {k: c for k, c in counts.items() if c}, draws, seed)
        acc_sorted = r.pop("sorted")
        rule("random_roles", r["mean"], m / n, expected=r["expected"], ci95=r["ci95"], draws=draws,
             share_at_least_learned=sum(a >= _mean(learned) for a in acc_sorted) / draws)

    curve = confidence_curve(rows) if "verifier" in kinds else None
    if curve is not None:
        order = confidence_order(rows)
        first_m = set(order[:m])
        conf_m = gated_outcomes(rows, [i in first_m for i in range(n)])
        rule("confidence_matched", curve[m], m / n)
        out["tests"]["learned - confidence_matched"] = compare(conf_m, learned, n_boot, seed)
        best_k = max(range(n + 1), key=lambda k: (curve[k], -k))
        rule("confidence_best_in_pool", curve[best_k], best_k / n, note="threshold chosen on this pool (upper bound)")
        out["confidence_curve"] = [{"calls": k / n, "accuracy": a,
                                    "random_expected": curve[0] + k / n * (curve[n] - curve[0])}
                                   for k, a in enumerate(curve)]
        if dev_records is not None:
            dev = table(dev_records)
            dev_curve = confidence_curve(dev)
            # questions that are also in the dev grid would make the "dev threshold" partly in-sample: leave them out
            shared = {row["qid"] for row in dev} & {row["qid"] for row in rows}
            keep = [i for i, row in enumerate(rows) if row["qid"] not in shared]
            if shared:
                out["dev_overlap"] = {"shared": n - len(keep), "kept": len(keep)}
            if dev_curve is not None and keep:
                sub, sub_learned = [rows[i] for i in keep], [learned[i] for i in keep]
                dev_m = sum(row["action"] != "commit" for row in dev)
                dev_confs = sorted(row["conf"] for row in dev)
                # realisable thresholds only: cuts between two distinct dev confidences (and the two ends)
                cuts = [k for k in range(len(dev) + 1) if k in (0, len(dev)) or dev_confs[k - 1] < dev_confs[k]]
                dev_matched = min(cuts, key=lambda k: (abs(k - dev_m), k))
                dev_best = max(cuts, key=lambda k: (dev_curve[k], -k))
                for name, k in (("confidence_dev_matched", dev_matched), ("confidence_dev_best", dev_best)):
                    t = threshold_for_k(dev, k)
                    gated = [row["conf"] < t for row in sub]
                    outcome = gated_outcomes(sub, gated)
                    dev_gated = [row["conf"] < t for row in dev]
                    kw: Dict[str, Any] = {}
                    if shared:  # the learned rule on the same reduced set of questions, for the reader of the table
                        kw = {"n": len(keep), "dev_overlap_excluded": n - len(keep),
                              "learned_same_questions": {"accuracy": _mean(sub_learned),
                                                         "calls": _mean(row["action"] != "commit" for row in sub)}}
                    rule(name, _mean(outcome), sum(gated) / len(sub), threshold=t if math.isfinite(t) else None,
                         dev_calls=sum(dev_gated) / len(dev), dev_accuracy=_mean(gated_outcomes(dev, dev_gated)),
                         dev_policy_calls=dev_m / len(dev), **kw)
                    out["tests"][f"learned - {name}"] = compare(outcome, sub_learned, n_boot, seed)
    return out


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _p(p) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


LABELS = [
    ("candidate", "Candidate only (never call)"),
    ("learned", "Learned calling, first call only (policy's advisor, then commit)"),
    ("learned_selection_verifier", "Learned selection, Verifier only"),
    ("random_roles", "Random questions, same calls per advisor"),
    ("random_verifier", "Random questions, Verifier only"),
    ("confidence_matched", "Confidence-gated Verifier, same number of calls"),
    ("confidence_dev_matched", "Confidence-gated Verifier, dev threshold (matched call rate)"),
    ("confidence_dev_best", "Confidence-gated Verifier, dev threshold (best dev accuracy)"),
    ("confidence_best_in_pool", "Confidence-gated Verifier, best threshold on this pool"),
    ("always_verifier", "Always Verifier, with the candidate"),
    ("always_verifier_nocand", "Always Verifier, candidate removed"),
    ("always_reasoner", "Always Reasoner"),
    ("always_extractor", "Always Extractor"),
    ("always_null", "Revision turn on an empty Verifier output (no advisor call)"),
    ("oracle", "Oracle (best of candidate / one call)"),
]


ROOT_AGREEMENT_FLOOR = 0.95  # below it the first turn is the grammar's doing, not the model's own behaviour


def markdown(result: Dict[str, Any], title: str) -> str:
    agreement = result.get("root_agreement")
    forced_root = bool(agreement) and agreement["n_equal"] < ROOT_AGREEMENT_FLOOR * agreement["n"]
    who = "the grammar-constrained first action" if forced_root else "the learned policy's first action"
    lines = [f"### {title}", "",
             f"n = {result['n']}; {who} is a call on {result['policy_calls']} questions "
             f"({', '.join(f'{k} {v}' for k, v in result['policy_call_counts'].items())}). Every rule makes at most one "
             "call and is scored by the same forced-commit revision.", ""]
    if agreement:
        lines += [f"The grid's first turn equals unconstrained greedy decoding on {agreement['n_equal']}/{agreement['n']} "
                  "questions." + (" This manager does not follow the DRAFT / commit / call grammar on its own: the rows "
                                  "below describe the grammar-forced model, not its deployed behaviour, and the budget-"
                                  "matched rows use that forced call budget." if forced_root else ""), ""]
    if result.get("dev_overlap"):
        o = result["dev_overlap"]
        lines += [f"{o['shared']} of the {result['n']} questions are also in the dev grid; the two dev-threshold rows and "
                  f"their tests use the other {o['kept']}.", ""]
    lines += ["| Rule | Calls per question | Accuracy (%) | Note |", "|:--|--:|--:|:--|"]
    for key, label in LABELS:
        r = result["rules"].get(key)
        if r is None:
            continue
        if key == "learned" and forced_root:
            label = "Grammar-constrained first action, then commit (not the deployed policy)"
        note = ""
        if "ci95" in r:
            note = (f"95% of draws in [{_pct(r['ci95'][0])}, {_pct(r['ci95'][1])}]; "
                    f"{_p(r['share_at_least_learned'])} of draws ≥ learned")
            if "share_at_least_learned_selection_verifier" in r:
                note += f"; {_p(r['share_at_least_learned_selection_verifier'])} ≥ learned selection (Verifier only)"
        elif "threshold" in r:
            note = f"dev: {_pct(r['dev_accuracy'])} at {r['dev_calls']:.2f} calls"
            if "n" in r:
                same = r["learned_same_questions"]
                note += (f"; n = {r['n']} ({r['dev_overlap_excluded']} dev questions excluded; learned on these: "
                         f"{_pct(same['accuracy'])} at {same['calls']:.2f} calls)")
        elif "commit_forced" in r:
            note = (f"after {r['commit_forced']} of {result['policy_calls']} calls ({_pct(r['commit_forced_share'])}%) the "
                    "commit was forced (the model's own next step was not the commit)")
        elif "note" in r:
            note = r["note"]
        lines.append(f"| {label} | {r['calls']:.2f} | {_pct(r['accuracy'])} | {note} |")
    lines += ["", "| Paired test (first minus second) | Δ accuracy (pt) | 95% CI | p (bootstrap) | p (McNemar) |",
              "|:--|--:|:-:|--:|--:|"]
    for name, t in result["tests"].items():
        lines.append(f"| {name} | {100 * t['diff']:+.1f} | [{100 * t['ci95'][0]:+.1f}, {100 * t['ci95'][1]:+.1f}] | "
                     f"{_p(t['p'])} | {_p(t['mcnemar_p'])} |")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="grid_records.jsonl or counterfactual_records.jsonl")
    ap.add_argument("--dev", default=None, help="the same checkpoint's dev grid (fixes the confidence threshold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True, help="output file stem and table title")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    result = analyze(read_records(args.records), read_records(args.dev) if args.dev else None, draws=args.draws,
                     n_boot=args.n_boot, seed=args.seed)
    result.update(records=str(args.records), dev=args.dev, label=args.label)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (out / f"{args.label}.md").write_text(markdown(result, args.label), encoding="utf-8")
    if "confidence_curve" in result:
        with open(out / f"{args.label}_curve.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["calls_per_question", "confidence_gated_accuracy", "random_verifier_expected_accuracy"])
            for p in result["confidence_curve"]:
                w.writerow([f"{p['calls']:.6f}", f"{p['accuracy']:.6f}", f"{p['random_expected']:.6f}"])
    print(markdown(result, args.label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
