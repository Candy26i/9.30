#!/usr/bin/env python3
"""Statistics and paper-style tables for MCQ RSI runs (design §5).

Subcommands
  compare   paired bootstrap (10k) for accuracy and calls + McNemar between two evals on identical ids
  replay    matched-budget replay (paper §4.4): fix role and call count, resample examples 2,000 times
  agree     per-example candidate + tool-sequence agreement of two evals (round-0 parity vs a recorded eval)
  tables    per-round paper Tables 1, 2, 3, 7, 8 from a run directory, finals vs S_1 / between arms
            (paired bootstrap + McNemar) and matched-budget replay of every final on dev; CSV + markdown.
            A continuation run (``continue_from``) has no S_1 final: dynamic vs the controls only
  smoke-check  the automatic GPU-smoke pass checks (design §7.2): run complete, round-0 agreement of
            r1/S1_dev with the recorded paper eval (>= 48 of 50), FA-GRPO peak memory, an accept block
            on every GRPO dev eval; writes <run>/smoke_check.json, exit 1 on failure

An "eval" argument is a ``manager_tool_eval.jsonl`` / ``manager_forced_*.jsonl`` file or a stage
directory holding one. Only the standard library is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

KINDS = ("extractor", "reasoner", "verifier")


# ------------------------------------------------------------------------------- I/O

def eval_file(path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    for name in ("manager_tool_eval.jsonl",):
        if (path / name).is_file():
            return path / name
    forced = sorted(path.glob("manager_forced_*.jsonl"))
    if len(forced) == 1:
        return forced[0]
    raise FileNotFoundError(f"{path}: no eval records")


def load_eval(path) -> Dict[int, Dict[str, Any]]:
    with open(eval_file(path), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out = {int(r["example_id"]): r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"{path}: duplicate example ids")
    return out


def _paired(a: Dict[int, Dict[str, Any]], b: Dict[int, Dict[str, Any]]) -> List[int]:
    if set(a) != set(b):
        raise ValueError(f"evals cover different ids ({len(set(a) ^ set(b))} differ); paired tests need identical ids")
    return sorted(a)


# ------------------------------------------------------------------------- statistics

def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default) of an already sorted list."""
    if not sorted_values:
        raise ValueError("empty sample")
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> Dict[str, Any]:
    """Mean of b - a with a percentile 95% interval from ``n_boot`` paired resamples of the examples.

    ``p_two_sided`` = 2 * min(P*(diff <= 0), P*(diff >= 0)), capped at 1 (bootstrap sign test).
    """
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and equally long")
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    le = sum(x <= 0 for x in boots) / n_boot
    ge = sum(x >= 0 for x in boots) / n_boot
    return {"n": n, "mean_a": sum(map(float, a)) / n, "mean_b": sum(map(float, b)) / n, "diff": sum(diffs) / n,
            "ci95": [percentile(boots, 0.025), percentile(boots, 0.975)], "p_two_sided": min(1.0, 2 * min(le, ge)),
            "n_boot": n_boot, "seed": seed}


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> Dict[str, Any]:
    """McNemar on paired binary outcomes: exact two-sided binomial p and the continuity-corrected chi-square."""
    if len(a_correct) != len(b_correct):
        raise ValueError("paired outcomes must be equally long")
    only_a = sum(bool(x) and not bool(y) for x, y in zip(a_correct, b_correct))
    only_b = sum(bool(y) and not bool(x) for x, y in zip(a_correct, b_correct))
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "exact_p": 1.0, "chi2_cc": 0.0, "chi2_p": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    chi2 = (abs(only_a - only_b) - 1) ** 2 / n
    chi2_p = math.erfc(math.sqrt(chi2 / 2))  # survival of chi-square with 1 dof
    return {"only_a": only_a, "only_b": only_b, "exact_p": min(1.0, 2 * tail), "chi2_cc": chi2, "chi2_p": chi2_p}


def compare(a: Dict[int, Dict[str, Any]], b: Dict[int, Dict[str, Any]], n_boot: int = 10_000,
            seed: int = 0) -> Dict[str, Any]:
    ids = _paired(a, b)
    acc_a, acc_b = [bool(a[i]["correct"]) for i in ids], [bool(b[i]["correct"]) for i in ids]
    calls = lambda r: float(r.get("tool_calls", 0))
    return {"n": len(ids), "accuracy": paired_bootstrap(acc_a, acc_b, n_boot, seed),
            "calls": paired_bootstrap([calls(a[i]) for i in ids], [calls(b[i]) for i in ids], n_boot, seed),
            "mcnemar": mcnemar(acc_a, acc_b)}


def first_role(record: Dict[str, Any]) -> Optional[str]:
    names = record.get("tool_names_called") or []
    if not names:
        return None
    name = str(names[0])
    return name[:-5] if name.endswith("_tool") else name


def matched_budget_replay(policy: Dict[int, Dict[str, Any]], forced: Dict[str, Dict[int, Dict[str, Any]]],
                          n_resamples: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Paper §4.4: does the policy call on the right examples, at a fixed role and call count?

    Each called example is credited with the forced-``role`` outcome of its first role (the
    forced eval of that role on the same ids); every other example keeps its candidate
    answer. The random baseline gives the same number of examples per role to uniformly
    drawn disjoint example sets, ``n_resamples`` times. Called examples whose role has no
    forced eval keep their policy outcome in both (``fixed``) and are not resampled.
    """
    ids = sorted(policy)
    for role, rows in forced.items():
        if set(rows) != set(ids):
            raise ValueError(f"forced {role} eval covers different ids")
    cand = {i: bool(policy[i].get("initial_draft_correct")) for i in ids}
    roles = {i: first_role(policy[i]) for i in ids}
    fixed = [i for i in ids if roles[i] is not None and roles[i] not in forced]
    free = [i for i in ids if i not in set(fixed)]
    counts = {r: sum(roles[i] == r for i in free) for r in forced}
    fixed_correct = sum(bool(policy[i]["correct"]) for i in fixed)

    def score(assign: Dict[int, str]) -> float:
        total = fixed_correct
        for i in free:
            role = assign.get(i)
            total += bool(forced[role][i]["correct"]) if role else cand[i]
        return total / len(ids)

    observed = score({i: roles[i] for i in free if roles[i] is not None})
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        order = free[:]
        rng.shuffle(order)
        assign, pos = {}, 0
        for role in sorted(counts):
            for i in order[pos:pos + counts[role]]:
                assign[i] = role
            pos += counts[role]
        samples.append(score(assign))
    samples.sort()
    return {"n": len(ids), "role_counts": counts, "fixed_unmatched": len(fixed), "policy_replay_accuracy": observed,
            "policy_accuracy": sum(bool(policy[i]["correct"]) for i in ids) / len(ids),
            "random_mean": sum(samples) / len(samples),
            "random_ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
            "p_random_ge_policy": sum(s >= observed for s in samples) / len(samples),
            "above_random_interval": observed > percentile(samples, 0.975), "n_resamples": n_resamples, "seed": seed}


def agreement(a: Dict[int, Dict[str, Any]], b: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-example agreement of candidate (initial draft) and tool sequence (design §7.2 step 3)."""
    ids = sorted(set(a) & set(b))
    same = [i for i in ids if a[i].get("initial_draft") == b[i].get("initial_draft")
            and list(a[i].get("tool_names_called") or []) == list(b[i].get("tool_names_called") or [])]
    return {"n_common": len(ids), "agree": len(same), "rate": len(same) / max(1, len(ids)),
            "disagree_ids": [i for i in ids if i not in set(same)][:50]}


# ------------------------------------------------------------------------------- tables

def _read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(x, digits=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(lines + ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows])


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def run_tables(run_dir, n_boot: int = 10_000, n_replay: int = 2000, seed: int = 0) -> Dict[str, Any]:
    root = Path(run_dir)
    report = _read_json(root / "report.json")
    tables: Dict[str, Dict[str, Any]] = {}
    tables["table1_oracle_by_depth"] = {
        "headers": ["stage", "round", "n", "commit A0", "best 1 call A1", "best measured AD", "Gain@1", "unsolved"],
        "rows": [[c["stage"], c["round"], c["n"], 100 * c["commit_A0"], 100 * c["best_one_call_A1"],
                  100 * c["best_measured_AD"], c["gain_at_1"], c["n_unsolved"]] for c in report["collections"]]}
    tables["table2_net_marginal_value"] = {
        "headers": ["stage", "round", "extractor 100Δ", "reasoner 100Δ", "verifier 100Δ"],
        "rows": [[c["stage"], c["round"], *[c["net_marginal_pp"].get(k) for k in KINDS]] for c in report["collections"]]}
    tables["table3_test"] = {
        "headers": ["stage", "label", "pool", "n", "candidate", "MARGENT", "gain pp", "calls"],
        "rows": [[f["stage"], f["label"], f["pool"], f["n"], f["candidate"], f["accuracy"], f["gain_pp"],
                  f["calls_per_example"]] for f in report["finals"] if not f["forced"]]}
    tables["table7_selectivity"] = {
        "headers": ["stage", "round", "candidate", "margin", "call gap"],
        "rows": [[d["stage"], d["round"], d["initial_draft_accuracy"], d["margin"], d["call_gap"]] for d in report["dev"]]}
    tables["table8_outcomes"] = {
        "headers": ["stage", "round", "calls", "correction", "corruption"],
        "rows": [[d["stage"], d["round"], d["calls_per_example"], d["correction_rate"], d["corruption_rate"]]
                 for d in report["dev"]]}
    stats: Dict[str, Any] = {"comparisons": [], "replay": []}
    finals_path = root / "final" / "finals.json"
    if finals_path.is_file():
        finals = _read_json(finals_path)
        stages = {s["name"]: s for s in finals["stages"]}
        tests = [s for s in finals["stages"] if s["params"].get("forced") is None]
        pools = sorted({s["params"]["pool"] for s in tests})
        for pool in pools:
            evals = {s["params"]["label"]: root / s["name"] for s in tests if s["params"]["pool"] == pool
                     and (root / s["name"] / "manager_tool_eval.jsonl").is_file()}
            labels = sorted(evals, key=lambda x: (x != "S_1", x))
            for i, la in enumerate(labels):
                for lb in labels[i + 1:]:
                    if la != "S_1" and "dynamic" not in (la, lb):
                        continue  # vs S_1 and dynamic vs controls only (pre-registered comparisons)
                    res = compare(load_eval(evals[la]), load_eval(evals[lb]), n_boot, seed)
                    stats["comparisons"].append({"pool": pool, "a": la, "b": lb, **res})
        for label, f in finals["finals"].items():
            dev_result = None
            if f["ref"].startswith("decision:"):
                stage = f["ref"][len("decision:"):].partition("#")[0]
                dev_result = _read_json(root / stage / "decision.json")["dev_result"]
            elif (root / "r1/S1_dev/mcq_rsi_eval.json").is_file():
                dev_result = str(root / "r1/S1_dev/mcq_rsi_eval.json")
            if not dev_result:
                continue
            forced, excluded = {}, {}
            for s in stages.values():
                tools = s["params"].get("forced")
                if s["params"]["label"] == label and tools and "," not in tools and (root / s["name"]).is_dir():
                    final_json = root / s["name"] / "final.json"
                    if final_json.is_file() and _read_json(final_json).get("broken"):
                        # Too many invalid answers: its outcomes would bias the random baseline; those examples
                        # keep their policy outcome (fixed) instead.
                        excluded[tools] = _read_json(final_json)["metrics"].get("valid_answer_rate")
                        continue
                    try:
                        forced[tools] = load_eval(root / s["name"])
                    except FileNotFoundError:
                        pass
            if forced:
                res = matched_budget_replay(load_eval(Path(dev_result).parent), forced, n_replay, seed)
                stats["replay"].append({"label": label, "dev_result": dev_result, **res,
                                        "excluded_broken_roles": excluded})
    return {"tables": tables, "stats": stats}


def write_tables(result: Dict[str, Any], out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# MCQ RSI tables", ""]
    for name, t in result["tables"].items():
        write_csv(out / f"{name}.csv", t["headers"], t["rows"])
        md += [f"## {name}", "", markdown_table(t["headers"], t["rows"]), ""]
    comps = result["stats"]["comparisons"]
    if comps:
        headers = ["pool", "a", "b", "n", "Δacc", "Δacc 95% CI", "p", "Δcalls", "Δcalls 95% CI", "McNemar exact p"]
        rows = [[c["pool"], c["a"], c["b"], c["n"], c["accuracy"]["diff"],
                 "[{:.3f}, {:.3f}]".format(*c["accuracy"]["ci95"]), c["accuracy"]["p_two_sided"], c["calls"]["diff"],
                 "[{:.3f}, {:.3f}]".format(*c["calls"]["ci95"]), c["mcnemar"]["exact_p"]] for c in comps]
        write_csv(out / "paired_tests.csv", headers, rows)
        md += ["## Paired tests (b - a)", "", markdown_table(headers, rows), ""]
    reps = result["stats"]["replay"]
    if reps:
        headers = ["label", "n", "roles", "fixed", "policy replay acc", "random 95% interval", "p", "above interval",
                   "excluded (broken forced eval: valid rate)"]
        rows = [[r["label"], r["n"], json.dumps(r["role_counts"]), r["fixed_unmatched"], r["policy_replay_accuracy"],
                 "[{:.3f}, {:.3f}]".format(*r["random_ci95"]), r["p_random_ge_policy"], r["above_random_interval"],
                 json.dumps(r.get("excluded_broken_roles") or {})]
                for r in reps]
        write_csv(out / "matched_budget_replay.csv", headers, rows)
        md += ["## Matched-budget replay (dev)", "", markdown_table(headers, rows), ""]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    (out / "analysis.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return out / "tables.md"


# --------------------------------------------------------------------------- smoke check

def extract_member(archive, member: str, dest_dir) -> Path:
    """One member of a (gzip) tarball, matched with or without a leading ``./``, into ``dest_dir/member``."""
    dest = Path(dest_dir) / member
    if dest.is_file():
        return dest
    with tarfile.open(archive, "r:*") as tar:
        for info in tar:
            name = info.name[2:] if info.name.startswith("./") else info.name
            if name == member and info.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                with tar.extractfile(info) as fin, open(tmp, "wb") as fout:
                    shutil.copyfileobj(fin, fout)
                tmp.replace(dest)
                return dest
    raise FileNotFoundError(f"{archive}: no member {member}")


def smoke_check(run_dir, recorded, min_agree: int = 48, max_peak_gb: float = 60.0) -> Dict[str, Any]:
    """Design §7.2 pass checks that need no operator judgement (the rest are listed under ``manual``)."""
    root = Path(run_dir)
    report = _read_json(root / "report.json")
    checks = [{"check": "every planned stage complete", "passed": bool(report["complete"]),
               "detail": report["pending"][:10]}]
    ag = agreement(load_eval(root / "r1" / "S1_dev"), load_eval(recorded))
    checks.append({"check": f"round-0 agreement (candidate + tool sequence) >= {min_agree} of the recorded paper eval",
                   "passed": ag["n_common"] >= min_agree and ag["agree"] >= min_agree,
                   "detail": {k: ag[k] for k in ("n_common", "agree", "rate", "disagree_ids")}})
    peaks = {str(f.parent.relative_to(root)): _read_json(f).get("peak_memory_gb")
             for f in sorted(root.glob("**/controller_stage.json"))}
    checks.append({"check": f"FA-GRPO peak memory < {max_peak_gb} GB (torch.cuda.max_memory_allocated)",
                   "passed": bool(peaks) and all(v is not None and v < max_peak_gb for v in peaks.values()),
                   "detail": peaks})
    grpo_devs = [d for d in report["decisions"] if d.get("role") == "grpo"]
    checks.append({"check": "an accept block on every GRPO dev eval", "passed": bool(grpo_devs) and all(
        "accept" in d for d in grpo_devs), "detail": {d["stage"]: (d.get("accept") or {}).get("decision") for d in grpo_devs}})
    gated = [d["stage"] for d in report["dev"] if d.get("gate")]
    checks.append({"check": "no eval-gate failure on any dev eval", "passed": not gated, "detail": gated})
    return {"passed": all(c["passed"] for c in checks), "checks": checks, "run_dir": str(root),
            "manual": ["§5 evidence lines and preflight PASS for medqa",
                       "forced rollback: rerun one GRPO stage with guard_max_kl_dec -1; it must end at step "
                       "guard_every with selected_step 0 (runbook §7)",
                       "copy the stage wall times (status) into runbook §16"]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("replay")
    p.add_argument("--policy", required=True)
    p.add_argument("--forced", action="append", default=[], metavar="ROLE=EVAL")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("agree")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None, help="default: <run-dir>/analysis")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--n-replay", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("smoke-check")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--recorded", required=True, help="recorded paper manager_tool_eval.jsonl, or a .tgz with --member")
    p.add_argument("--member", default=None, help="member of --recorded (a tarball) to extract into --extract-to")
    p.add_argument("--extract-to", default=None, help="default: <run-dir>/recorded")
    p.add_argument("--min-agree", type=int, default=48)
    p.add_argument("--max-peak-gb", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.command == "smoke-check":
        recorded = args.recorded
        if args.member:
            recorded = extract_member(args.recorded, args.member, args.extract_to or Path(args.run_dir) / "recorded")
        result = smoke_check(args.run_dir, recorded, args.min_agree, args.max_peak_gb)
        (Path(args.run_dir) / "smoke_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        for c in result["checks"]:
            print(f"[smoke-check] {'PASS' if c['passed'] else 'FAIL'}  {c['check']}: {json.dumps(c['detail'])[:300]}")
        for m in result["manual"]:
            print(f"[smoke-check] MANUAL {m}")
        print(f"[smoke-check] {'PASSED' if result['passed'] else 'FAILED'} -> {Path(args.run_dir) / 'smoke_check.json'}")
        return 0 if result["passed"] else 1
    if args.command == "compare":
        result = compare(load_eval(args.a), load_eval(args.b), args.n_boot, args.seed)
    elif args.command == "replay":
        forced = {}
        for item in args.forced:
            role, _, path = item.partition("=")
            forced[role] = load_eval(path)
        result = matched_budget_replay(load_eval(args.policy), forced, args.n, args.seed)
    elif args.command == "agree":
        result = agreement(load_eval(args.a), load_eval(args.b))
    else:
        result = run_tables(args.run_dir, args.n_boot, args.n_replay, args.seed)
        path = write_tables(result, args.out or Path(args.run_dir) / "analysis")
        print(path.read_text(encoding="utf-8"))
        return 0
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
