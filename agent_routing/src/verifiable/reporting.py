"""Recompute paper tables/figures from observed question-level records only."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
import itertools
from pathlib import Path
import statistics

from ..utils.io import read_jsonl, write_jsonl
from .telemetry import atomic_json
from .analysis import conditional_metrics, state_of


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p, d = successes / n, 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., mid - radius), min(1., mid + radius)


def paired_stats(before, after, seed=42, samples=4000):
    import numpy as np
    if len(before) != len(after) or not before:
        raise ValueError("Paired statistics require nonempty aligned question sets")
    if samples < 1 or not all(value in (0, 1, False, True) for value in before + after):
        raise ValueError("Paired accuracy statistics require binary outcomes and positive bootstrap samples")
    delta = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, samples, 200):
        boot.extend(delta[rng.integers(0, len(delta), (min(200, samples - start), len(delta)))].mean(axis=1))
    gained, lost = int((delta > 0).sum()), int((delta < 0).sum())
    discordant = gained + lost
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1)) / 2 ** discordant)
    return {"newly_solved": gained, "regressed": lost, "delta_pp": float(delta.mean() * 100),
            "delta_ci_low_pp": float(np.quantile(boot, .025) * 100),
            "delta_ci_high_pp": float(np.quantile(boot, .975) * 100), "mcnemar_exact_p": p}


def read_optional(path):
    return read_jsonl(str(path)) if Path(path).exists() else []


def coverage(r):
    return bool(r["direct_correct"] or any(b["correct"] for b in r["branches"]))


def metrics(records):
    n = len(records)
    if not n:
        raise ValueError("Cannot report an empty evaluation")
    result = {"n": n}
    for key, fn, required in [
        ("independent", lambda r: r["direct_correct"], "direct_correct"),
        ("policy", lambda r: r["policy"]["correct"], "policy"),
        ("search", coverage, "branches"),
        ("self_continue", lambda r: r["self_continue_correct"], "self_continue_correct")]:
        if all(required in r for r in records):
            successes = sum(bool(fn(r)) for r in records)
            low, high = wilson(successes, n)
            result.update({f"{key}_correct": successes, f"{key}_pct": successes / n * 100,
                           f"{key}_ci_low_pct": low * 100, f"{key}_ci_high_pct": high * 100})
    if all("policy" in r for r in records):
        result["mean_calls"] = statistics.mean(r["policy"]["calls"] for r in records)
        result["policy_valid_pct"] = 100 * statistics.mean(r["policy"]["valid"] for r in records)
        successes = sum(bool(r["direct_correct"]) for r in records)
        rescued = sum(not r["direct_correct"] and r["policy"]["correct"] for r in records)
        harmed = sum(r["direct_correct"] and not r["policy"]["correct"] for r in records)
        result.update(policy_rescued_questions=rescued, policy_harmed_questions=harmed,
                      policy_rescue_rate_pct=100 * rescued / (n - successes) if n > successes else None,
                      policy_harm_rate_pct=100 * harmed / successes if successes else None)
        # Deployment cost is logical tokens INCLUDING the initial draft. It must
        # not become artificially cheap through caches primed by diagnostic CF.
        if all(r.get("costs") and "costs" in r["policy"] for r in records):
            result["mean_policy_logical_tokens"] = statistics.mean(
                sum(c["prompt_tokens"] + c["completion_tokens"] for c in [r["costs"][0]] + r["policy"]["costs"])
                for r in records)
    for field in ("direct_valid", "direct_truncated"):
        if all(field in r for r in records):
            result[field + "_pct"] = 100 * statistics.mean(r[field] for r in records)
    if all("branches" in r for r in records):
        rescue = sum(not r["direct_correct"] and coverage(r) for r in records)
        failures = sum(not r["direct_correct"] for r in records)
        result.update(rescued_questions=rescue, direct_failures=failures,
                      rescue_rate_pct=100 * rescue / failures if failures else None,
                      branch_count=sum(len(r["branches"]) for r in records))
    result.update(conditional_metrics(records))
    return result


def aligned(before, after):
    a, b = ({r["question_hash"]: r for r in rows} for rows in (before, after))
    if len(a) != len(before) or len(b) != len(after) or set(a) != set(b):
        raise ValueError("Duplicate or mismatched question identities in paired comparison")
    keys = sorted(a)
    for key in keys:
        for field in ("ground_truth", "question", "context", "split", "benchmark_name"):
            if field in a[key] and field in b[key] and a[key][field] != b[key][field]:
                raise ValueError(f"Paired question metadata differs: {key}/{field}")
    return keys, [a[k] for k in keys], [b[k] for k in keys]


def transitions(before, after, meta, reference):
    keys, a, b = aligned(before, after)
    rows, questions = [], []
    for name, fn in [("independent", lambda r: r["direct_correct"]),
                     ("policy", lambda r: r["policy"]["correct"])] + (
                         [("search", coverage)] if all("branches" in r for r in a + b) else []):
        av, bv = [bool(fn(r)) for r in a], [bool(fn(r)) for r in b]
        rows.append({**meta, "reference": reference, "metric": name, "n": len(a), **paired_stats(av, bv)})
    for key, old, new in zip(keys, a, b):
        item = {**meta, "reference": reference, "question_hash": key,
                "before_independent": old["direct_correct"], "after_independent": new["direct_correct"],
                "before_calls": old["policy"]["calls"], "after_calls": new["policy"]["calls"],
                "before_state": state_of(old) if "branches" in old else None,
                "after_state": state_of(new) if "branches" in new else None,
                "newly_solved_independent": not old["direct_correct"] and new["direct_correct"],
                "regressed_independent": old["direct_correct"] and not new["direct_correct"]}
        if "branches" in old:
            item.update(previously_rescued=not old["direct_correct"] and coverage(old),
                        previously_rescued_now_independent=not old["direct_correct"] and coverage(old) and new["direct_correct"],
                        outside_initial_search_now_independent=not coverage(old) and new["direct_correct"])
        questions.append(item)
    return rows, questions


def stage_cost(path):
    events = read_optional(path / "events.jsonl")
    ledger = read_optional(path / "usage.jsonl")
    starts = {e["attempt"] for e in events if e["event"] == "started"}
    ends = {e["attempt"] for e in events if e["event"] in {"completed", "failed", "interrupted"}}
    usage_incomplete = any(e["event"] == "usage_tail_recovered" for e in events)
    clean = bool(starts) and starts == ends and not usage_incomplete and all(e["event"] != "failed" and e["event"] != "interrupted" for e in events)
    out = {"stage_path": str(path), "observed_wall_seconds": sum(e.get("wall_seconds", 0) for e in events),
           "attempts": len(starts), "accounting_complete": clean and bool(ledger),
           "usage_incomplete": usage_incomplete,
           "sft_processed_tokens": sum(r.get("input_tokens", 0) for r in ledger if r["role"] == "sft_train"),
           "sft_supervised_tokens": sum(r.get("supervised_tokens", 0) for r in ledger if r["role"] == "sft_train")}
    for role, match in [("manager", {"manager", "manager_rl"}), ("advisor", {"advisor"})]:
        for kind in ("prompt", "completion"):
            out[f"{role}_actual_{kind}_tokens"] = sum(r.get(f"actual_{kind}_tokens", r.get(f"{kind}_tokens", 0))
                for r in ledger if r["role"] in match)
    out["actual_generation_tokens"] = out["manager_actual_completion_tokens"] + out["advisor_actual_completion_tokens"]
    return out


def table(output, name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    def escape(value):
        if value is None:
            return "--"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    text = "\\begin{tabular}{" + "l" * len(fields) + "}\n\\hline\n"
    text += " & ".join(map(escape, fields)) + " \\\\\n\\hline\n"
    text += "".join(" & ".join(escape(r.get(k)) for k in fields) + " \\\\\n" for r in rows)
    (output / (name + ".tex")).write_text(text + "\\hline\n\\end{tabular}\n")


def generate_report(run_dirs, output, demo=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    request = {"run_dirs": [str(Path(p).resolve()) for p in run_dirs], "demo": demo}
    request_path = out / "report_request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Report input runs changed; use another output directory")
    if request_path.exists():
        # Remove only this reporter's previous exports; never leave an obsolete
        # figure behind when a refreshed report has no supporting observations.
        names = ("main_results", "development", "paired_changes", "costs", "training_diagnostics", "seed_summary", "paper_main", "paper_mechanism", "arm_comparisons", "arm_summary", "delegation_behavior")
        figures = ("fig1_development", "fig2_internalization", "fig3_external_tests", "fig4_costs", "fig5_rl_training", "fig6_delegation")
        for name, extensions in [(n, ("csv", "tex")) for n in names] + [(n, ("pdf", "png")) for n in figures]:
            for extension in extensions:
                (out / f"{name}.{extension}").unlink(missing_ok=True)
    elif any(out.iterdir()):
        raise ValueError("Report output is not empty; choose a fresh directory")
    atomic_json(request_path, request)
    main, dev, changes, questions, costs, training, audits = [], [], [], [], [], [], []
    warnings, hashes, endpoints = [], {}, []
    run_names = [Path(p).resolve().name for p in run_dirs]
    if len(set(run_names)) != len(run_names):
        raise ValueError("Run directory names must be unique (include arm and seed)")
    for run_dir in run_dirs:
        root = Path(run_dir).resolve()
        for path in root.rglob("*"):
            # W&B evidence snapshots are copies, not experimental inputs. Their
            # hashes change with uploads and must not alter the paper audit.
            if {"tracking_exports", "wandb"} & set(path.relative_to(root).parts):
                continue
            if path.is_file() and (path.name in {"loop.json", "advisor_identity.json", "training_run.json", "usage.jsonl", "events.jsonl", "training_log.jsonl", "rollouts.jsonl"} or path.name.startswith("environment_")):
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[str(path)] = digest.hexdigest()
        run = json.loads((root / "loop.json").read_text())
        cfg = run["config"]
        protocol = {k: v for k, v in cfg.items() if k not in {"seed", "advisor_url"}}
        identity_path = root / "advisor_identity.json"
        protocol["advisor_identity"] = json.loads(identity_path.read_text()) if identity_path.exists() else None
        if not identity_path.exists():
            warnings.append(f"{root.name}: frozen advisor identity was not recorded")
        protocol["data_hashes"] = run["data_manifest"].get("sha256", {})
        protocol["harness"] = run.get("harness")
        protocol["initial_checkpoint"] = run.get("initial")
        protocol["rounds"] = run.get("rounds")
        group = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
        common = {"run": root.name, "arm": run["arm"], "seed": cfg["seed"], "protocol_group": group}
        audits.append({**common, "path": str(root), "config": cfg, "data_manifest": run["data_manifest"]})
        method_seconds, method_tokens, accounting = 0., 0, True
        initial, previous = None, None
        for step in run["plan"]:
            path = Path(step["output"])
            label = "/".join(path.relative_to(root).parts)
            complete = (path / ".stage_complete.json").exists()
            if not complete:
                warnings.append(f"{root.name}/{label}: stage incomplete; excluded from result tables")
            cost = {**common, "stage": label, "purpose": "diagnostic" if step["stage"] == "diagnose" else "method", **stage_cost(path)}
            costs.append(cost)
            if step["stage"] != "diagnose":
                method_seconds += cost["observed_wall_seconds"]
                method_tokens += cost["actual_generation_tokens"]
                accounting = accounting and cost["accounting_complete"]
                for log in read_optional(path / "training_log.jsonl"):
                    training.append({**common, "stage": label, **log})
                rollouts = read_optional(path / "rollouts.jsonl")
                if rollouts:
                    batches = defaultdict(list)
                    for r in rollouts:
                        if r.get("question_hash") and r.get("reward_batch") is not None:
                            batches[(r["reward_batch"], r["question_hash"])].append(r["correct"])
                    groups = [v for v in batches.values() if len(v) == cfg.get("num_generations", 4)]
                    audits.append({**common, "stage": label, "rollouts": len(rollouts), "complete_reward_groups": len(groups),
                                   "zero_variance_group_fraction": statistics.mean(len(set(v)) == 1 for v in groups) if groups else None})
                continue
            if not complete or not (path / "records.jsonl").exists():
                continue
            records = read_jsonl(str(path / "records.jsonl"))
            check_records(path, records, hashes, run["data_manifest"].get("sha256", {}).get("dev.jsonl"), cfg, run.get("harness"))
            if len(records) != run["data_manifest"]["counts"]["dev"]:
                raise ValueError(f"Development set is incomplete: {path}")
            if "max_depth" in cfg:
                expected = {seq for d in range(1, cfg["max_depth"] + 1)
                            for seq in itertools.permutations(("extractor", "reasoner", "verifier"), d)}
                if any(len(r["branches"]) != len(expected) or {tuple(b["sequence"]) for b in r["branches"]} != expected for r in records):
                    raise ValueError(f"Bounded counterfactual search is incomplete: {path}")
            position = 0 if label == "initial_dev" else int(path.parent.name.split("_")[-1]) * 2 - (1 if path.name == "sft_dev" else 0)
            meta = {**common, "checkpoint": label, "benchmark": "development"}
            result = {**meta, "position": position, **metrics(records),
                      "cumulative_method_hours": method_seconds / 3600,
                      "cumulative_method_generation_tokens": method_tokens, "accounting_complete": accounting}
            if initial is not None:
                baseline = initial[1]
                result.update(conditional_metrics(records, baseline))
                rescued = [r["question_hash"] for r in baseline if not r["direct_correct"] and coverage(r)]
                after = {r["question_hash"]: r for r in records}
                internalized = sum(after[k]["direct_correct"] for k in rescued)
                result.update(initial_rescued_n=len(rescued), rescued_now_independent_n=internalized,
                              internalization_pct=100 * internalized / len(rescued) if rescued else None,
                              outside_initial_search_now_independent_n=sum(not coverage(r) and after[r["question_hash"]]["direct_correct"] for r in baseline))
                for reference in [initial] + ([previous] if previous[0] != initial[0] else []):
                    paired, detail = transitions(reference[1], records, meta, reference[0])
                    changes.extend(paired)
                    questions.extend(detail)
            elif label == "initial_dev":
                initial = (label, records)
            previous = (label, records)
            dev.append(result)
            if step == [p for p in run["plan"] if p["stage"] == "diagnose"][-1]:
                endpoints.append({**common, "benchmark": "development", "records": records})
        for benchmark in ("aime2026", "beyondaime"):
            baseline = None
            for checkpoint in ("initial", "final"):
                path = root / "test" / checkpoint / benchmark
                if not all((path / name).exists() for name in ("summary.json", "records.jsonl", ".stage_complete.json")):
                    warnings.append(f"{root.name}/test/{checkpoint}/{benchmark}: missing external evaluation")
                    continue
                records = read_jsonl(str(path / "records.jsonl"))
                check_records(path, records, hashes, run["data_manifest"].get("sha256", {}).get(benchmark + ".jsonl"), cfg, run.get("harness"))
                if run["data_manifest"].get("sha256"):
                    final_training = [p for p in run["plan"] if p["stage"] in {"sft", "rl"}][-1]["output"]
                    expected_checkpoint = (run["initial"] or cfg["base_model"]) if checkpoint == "initial" else final_training
                    if json.loads((path / "summary.json").read_text()).get("checkpoint") != expected_checkpoint:
                        raise ValueError(f"External result checkpoint differs from its initial/final label: {path}")
                expected = run["data_manifest"]["counts"][benchmark]
                if len(records) != expected or any(r.get("benchmark_name") != benchmark or r.get("split") != "test" for r in records):
                    raise ValueError(f"Partial or mislabeled external test: {path}")
                meta = {**common, "checkpoint": checkpoint, "benchmark": benchmark}
                main.append({**meta, **metrics(records)})
                if checkpoint == "final":
                    endpoints.append({**common, "benchmark": benchmark, "records": records})
                costs.append({**common, "stage": f"test/{checkpoint}/{benchmark}", "purpose": "external_test", **stage_cost(path)})
                if checkpoint == "initial":
                    baseline = records
                elif baseline is not None:
                    paired, detail = transitions(baseline, records, meta, "initial")
                    changes.extend(paired)
                    questions.extend(detail)
        if not accounting:
            warnings.append(f"{root.name}: method cost accounting incomplete (legacy, interrupted or failed attempts); excluded from cost curves")
    grouped = defaultdict(list)
    for r in main:
        grouped[(r["arm"], r["benchmark"], r["checkpoint"], r["protocol_group"])].append(r)
    seeds = []
    for key, rows in grouped.items():
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate seeds in aggregate group {key}")
        item = dict(zip(("arm", "benchmark", "checkpoint", "protocol_group"), key))
        item["num_seeds"] = len(rows)
        for metric in ("independent_pct", "policy_pct", "mean_calls"):
            values = [r[metric] for r in rows]
            item[metric + "_mean"] = statistics.mean(values)
            item[metric + "_sd"] = statistics.stdev(values) if len(values) > 1 else None
        seeds.append(item)
    if any(r["num_seeds"] == 1 for r in seeds):
        warnings.append("Some results have one training seed: question-level intervals do not measure training-seed variability")
    if len({r["protocol_group"] for r in main + dev}) > 1:
        warnings.append("Multiple model/budget protocols present: do not interpret pooled comparisons as controlled ablations")
    arm_rows = []
    for target in endpoints:
        if target["arm"] != "dynamic_sft":
            continue
        for control in endpoints:
            if control["arm"] not in {"success_sft", "static_sft"}:
                continue
            if any(target[k] != control[k] for k in ("seed", "benchmark", "protocol_group")):
                continue
            _, before, after = aligned(control["records"], target["records"])
            meta = {k: target[k] for k in ("seed", "benchmark", "protocol_group")}
            meta.update(treatment=target["run"], control=control["run"], control_arm=control["arm"])
            for metric, getter in (("independent", lambda r: r["direct_correct"]),
                                   ("policy", lambda r: r["policy"]["correct"])):
                arm_rows.append({**meta, "metric": metric, **paired_stats(list(map(getter, before)), list(map(getter, after))),
                                 "mean_calls_delta": statistics.mean(a["policy"]["calls"] - b["policy"]["calls"] for a, b in zip(after, before))})
    table(out, "arm_comparisons", arm_rows)
    arm_groups = defaultdict(list)
    for row in arm_rows:
        arm_groups[(row["control_arm"], row["benchmark"], row["protocol_group"], row["metric"])].append(row)
    arm_summary = []
    for key, rows in arm_groups.items():
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError("Duplicate seed in paired arm comparison")
        deltas = [r["delta_pp"] for r in rows]
        arm_summary.append({**dict(zip(("control_arm", "benchmark", "protocol_group", "metric"), key)),
                            "num_seeds": len(rows), "delta_pp_mean": statistics.mean(deltas),
                            "delta_pp_sd": statistics.stdev(deltas) if len(rows) > 1 else None})
    table(out, "arm_summary", arm_summary)
    if not arm_rows:
        warnings.append("No matched dynamic_sft vs success_sft/static_sft comparison is available")
    behavior_keys = ("currently_", "initially_rescued_", "learned_subset_", "transition_")
    table(out, "delegation_behavior", [{k: v for k, v in r.items() if k in {"run", "checkpoint", "seed", "arm"} or k.startswith(behavior_keys)} for r in dev])
    for name, rows in [("main_results", main), ("development", dev), ("paired_changes", changes),
                       ("costs", costs), ("training_diagnostics", training), ("seed_summary", seeds)]:
        table(out, name, rows)
    # Compact manuscript tables; full precision and extra diagnostics remain in CSV.
    interval = lambda r, k: f"{r[k + '_pct']:.1f} [{r[k + '_ci_low_pct']:.1f}, {r[k + '_ci_high_pct']:.1f}]"
    compact = [{"Run": r["run"], "Checkpoint": r["checkpoint"], "Benchmark": r["benchmark"], "N": r["n"],
                "Independent (%)": interval(r, "independent"), "Policy (%)": interval(r, "policy"), "Calls": r["mean_calls"]} for r in main]
    table(out, "paper_main", compact)
    table(out, "paper_mechanism", [{"Run": r["run"], "Checkpoint": r["checkpoint"],
          "D (%)": r["independent_pct"], "P (%)": r["policy_pct"], "C (%)": r.get("search_pct"),
          "Internalized": f"{r['rescued_now_independent_n']}/{r['initial_rescued_n']}" if "initial_rescued_n" in r else None,
          "Beyond initial C": r.get("outside_initial_search_now_independent_n"),
          "Calls on learned (before)": r.get("learned_subset_before_mean_calls"),
          "Calls on learned (after)": r.get("initially_rescued_now_independent_mean_calls"),
          "Still rescuable policy accuracy": r.get("currently_rescuable_policy_accuracy")} for r in dev])
    write_jsonl(str(out / "paired_questions.jsonl"), questions)
    figures = plot(out, main, dev, changes, training, demo)
    manifest = {"demo": demo, "runs": audits, "warnings": warnings, "input_sha256": hashes,
                "figures": figures, "main_rows": len(main), "development_rows": len(dev),
                "statistics": "95% Wilson intervals over questions; paired bootstrap over questions (4000, seed=42); exact McNemar p, unadjusted exploratory",
                "cost_scope": "Observed method-stage wall time and inference tokens; excludes dev/test from method curves; not FLOPs, billed GPU hours or equal-compute evidence"}
    atomic_json(out / "report_manifest.json", manifest)
    title = "SYNTHETIC PIPELINE DEMO — NOT EXPERIMENTAL RESULTS" if demo else "MARGENT observed experiment report"
    text = f"# {title}\n\nMain table rows: {len(main)}. Development rows: {len(dev)}.\n\n"
    text += "Tables: CSV for analysis; paper_main.tex for the compact manuscript table. Figures: PDF and PNG.\n\n"
    text += "Intervals describe variation across questions, not training seeds. McNemar p values are exploratory and unadjusted for multiple comparisons.\n\n"
    text += "Missing experiments are excluded, never filled with zeros. Cost curves omit incomplete accounting. Generated-token counts omit training forward/backward computation.\n\n"
    text += "Warnings:\n\n" + "\n".join("- " + w for w in warnings) + "\n\n"
    for figure in figures:
        text += f"![{figure}]({figure}.png)\n\n"
    (out / "README.md").write_text(text, encoding="utf-8")
    return {"output": str(out.resolve()), "main_rows": len(main), "development_rows": len(dev), "figures": figures, "warnings": warnings}


def check_records(path, records, hashes, expected_data_sha=None, expected_config=None, expected_harness=None):
    if not records or len({r["question_hash"] for r in records}) != len(records):
        raise ValueError(f"Empty or duplicate records: {path}")
    saved = json.loads((path / "summary.json").read_text())
    if expected_data_sha:
        metadata = json.loads((path / "run.json").read_text())
        if expected_harness and (metadata.get("harness") != expected_harness or any(r.get("protocol_version") != expected_harness["protocol_version"] for r in records)):
            raise ValueError("Evaluation harness/protocol differs from frozen loop")
        if metadata["data_sha256"] != expected_data_sha or metadata["config"] != expected_config:
            raise ValueError(f"Evaluation data or protocol differs from frozen loop: {path}")
    if expected_config and records[0].get("protocol_version") == 2:
        from .answers import correct
        for row in records:
            if "ground_truth" not in row:
                raise ValueError("Protocol-v2 records must retain offline gold for grading audits")
            gold = row["ground_truth"]
            if row["direct_correct"] != bool(row["direct_valid"] and correct(row["direct_text"], gold)):
                raise ValueError("Independent raw answer disagrees with saved correctness")
            if row["policy"]["correct"] != bool(row["policy"]["valid"] and correct(row["policy"]["text"], gold)):
                raise ValueError("Policy raw answer disagrees with saved correctness")
            for branch in row.get("branches", []):
                if branch["correct"] != bool(branch["valid"] and correct(branch["text"], gold)):
                    raise ValueError("Branch raw answer disagrees with saved correctness")
    if saved["n"] != len(records):
        raise ValueError(f"Summary/record count mismatch: {path}")
    for k, v in metrics(records).items():
        old = {"independent_pct": "independent_accuracy", "policy_pct": "policy_accuracy", "search_pct": "delegation_search_coverage"}.get(k)
        if old in saved and not math.isclose(v / 100, saved[old], abs_tol=1e-10):
            raise ValueError(f"Saved metrics disagree with raw outcomes: {path}/{old}")
    for name in ("records.jsonl", "summary.json", "run.json"):
        source = path / name
        if source.exists():
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()


def plot(out, main, dev, changes, training, demo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    figures = []
    def save(fig, name):
        if demo:
            fig.text(.5, .01, "SYNTHETIC DEMO — NOT EXPERIMENTAL RESULTS", ha="center", color="#aa2222", weight="bold")
        fig.tight_layout(rect=(0, .04, 1, 1))
        for extension in ("pdf", "png"):
            fig.savefig(out / f"{name}.{extension}", bbox_inches="tight")
        plt.close(fig)
        figures.append(name)
    if dev:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for ax, metric, title in zip(axes, ("independent", "policy", "search"), ("Independent solving", "Deployed policy", "Bounded search coverage")):
            for run in sorted({r["run"] for r in dev}):
                rows = sorted([r for r in dev if r["run"] == run], key=lambda r: r["position"])
                ax.plot([r["position"] for r in rows], [r[metric + "_pct"] for r in rows], marker="o", label=run)
            positions = sorted({r["position"] for r in dev})
            ax.set_xticks(positions, ["Initial" if p == 0 else f"R{(p+1)//2}\n{'SFT' if p % 2 else 'RL'}" for p in positions])
            ax.set(title=title, ylabel="Correct (%)", ylim=(0, 100))
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        save(fig, "fig1_development")
        last = [max([r for r in dev if r["run"] == run], key=lambda r: r["position"]) for run in sorted({r["run"] for r in dev})]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for i, r in enumerate(last):
            pair = next((c for c in changes if c["run"] == r["run"] and c["checkpoint"] == r["checkpoint"]
                         and c["reference"] == "initial_dev" and c["metric"] == "independent"), None)
            if pair:
                axes[0].bar(i - .18, pair["newly_solved"], .35, color="#287d8e", label="Newly solved" if i == 0 else None)
                axes[0].bar(i + .18, -pair["regressed"], .35, color="#d6784f", label="Regressed" if i == 0 else None)
            if r.get("initial_rescued_n"):
                low, high = wilson(r["rescued_now_independent_n"], r["initial_rescued_n"])
                value = r["internalization_pct"]
                axes[1].errorbar(i, value, yerr=[[max(0, value-low*100)], [max(0, high*100-value)]], fmt="o", capsize=4)
                axes[1].annotate(f"{r['rescued_now_independent_n']}/{r['initial_rescued_n']}", (i, value), xytext=(5, 5), textcoords="offset points")
        for ax in axes:
            ax.set_xticks(range(len(last)), [r["run"] for r in last], rotation=20, ha="right")
            ax.grid(axis="y", alpha=.2)
        axes[0].axhline(0, color="black", linewidth=.6)
        axes[0].set(title="Independent changes vs initial (dev)", ylabel="Questions")
        axes[0].legend(fontsize=8)
        axes[1].set(title="Initially rescued → independently solved", ylabel="Conditional rate (%)", ylim=(0, 105))
        save(fig, "fig2_internalization")
    if dev and any("currently_independent_call_rate" in r for r in dev):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        for key, title, ax in zip(("currently_independent_call_rate", "initially_rescued_now_independent_mean_calls", "currently_rescuable_policy_accuracy"),
                                 ("Call rate on independently solved questions", "Mean calls on newly internalized questions", "Policy accuracy on currently rescuable questions"), axes):
            for run in sorted({r["run"] for r in dev}):
                rows = sorted([r for r in dev if r["run"] == run], key=lambda r: r["position"])
                ax.plot([r["position"] for r in rows], [r.get(key) if r.get(key) is not None else float("nan") for r in rows], marker="o", label=run)
            ax.set(title=title, xlabel="Checkpoint position")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        save(fig, "fig6_delegation")
    if main:
        benchmarks = sorted({r["benchmark"] for r in main})
        fig, axes = plt.subplots(1, len(benchmarks), figsize=(6 * len(benchmarks), 4), squeeze=False)
        for ax, benchmark in zip(axes[0], benchmarks):
            rows = [r for r in main if r["benchmark"] == benchmark]
            for shift, metric, color in [(-.18, "independent", "#287d8e"), (.18, "policy", "#d6784f")]:
                values = [r[metric + "_pct"] for r in rows]
                errors = [[max(0, v - r[metric + "_ci_low_pct"]) for v, r in zip(values, rows)],
                          [max(0, r[metric + "_ci_high_pct"] - v) for v, r in zip(values, rows)]]
                ax.bar([i + shift for i in range(len(rows))], values, .35, yerr=errors, capsize=2, label=metric, color=color)
            ax.set_xticks(range(len(rows)), [r["run"] + "\n" + r["checkpoint"] for r in rows], rotation=25, ha="right", fontsize=7)
            ax.set(title=benchmark + " (95% question-level Wilson CI)", ylabel="Correct (%)", ylim=(0, 105))
            ax.legend()
        save(fig, "fig3_external_tests")
    if dev or main:
        benchmarks = sorted({r["benchmark"] for r in main})
        fig, axes = plt.subplots(1, 1 + max(1, len(benchmarks)), figsize=(5 * (1 + max(1, len(benchmarks))), 3.8))
        for run in sorted({r["run"] for r in dev}):
            rows = sorted([r for r in dev if r["run"] == run and r["accounting_complete"]], key=lambda r: r["position"])
            if len(rows) >= 2:
                axes[0].plot([r["cumulative_method_hours"] for r in rows], [r["policy_pct"] for r in rows], marker="o", label=run)
        for ax, benchmark in zip(axes[1:], benchmarks):
            for run in sorted({r["run"] for r in main}):
                rows = sorted([r for r in main if r["run"] == run and r["benchmark"] == benchmark], key=lambda r: r["checkpoint"] == "final")
                if rows:
                    line, = ax.plot([r["mean_calls"] for r in rows], [r["policy_pct"] for r in rows], label=run, alpha=.8)
                    for r in rows:
                        ax.scatter(r["mean_calls"], r["policy_pct"], marker="o" if r["checkpoint"] == "final" else "x", color=line.get_color())
            ax.set(xlabel="Mean deployed advisor calls", ylabel="Policy correct (%)", title=benchmark + " (initial × → final ○)")
            ax.legend(fontsize=7)
        axes[0].set(xlabel="Observed method-stage hours (excludes dev/test)", ylabel="Dev policy correct (%)", title="Time–accuracy (not FLOPs or billed GPU hours)")
        if axes[0].lines:
            axes[0].legend(fontsize=7)
        else:
            axes[0].text(.5, .5, "Complete cost accounting unavailable", ha="center", transform=axes[0].transAxes)
        if not benchmarks:
            axes[1].text(.5, .5, "External tests unavailable", ha="center", transform=axes[1].transAxes)
        for ax in axes:
            ax.grid(alpha=.2)
        save(fig, "fig4_costs")
    logs = [r for r in training if "/rl" in r["stage"] and "reward" in r]
    if logs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for key in sorted({(r["run"], r["stage"], str(r.get("attempt"))) for r in logs}):
            rows = [r for r in logs if (r["run"], r["stage"], str(r.get("attempt"))) == key]
            axes[0].plot([r["step"] for r in rows], [r["reward"] for r in rows], label="/".join(key[:2]))
            zero = [r for r in rows if "frac_reward_zero_std" in r]
            if zero:
                axes[1].plot([r["step"] for r in zero], [r["frac_reward_zero_std"] for r in zero], label="/".join(key[:2]))
        axes[0].set(title="RL reward", xlabel="Optimizer step", ylabel="Mean reward")
        axes[1].set(title="RL groups with zero reward variance", xlabel="Optimizer step", ylabel="Fraction", ylim=(0, 1.05))
        axes[0].legend(fontsize=7)
        save(fig, "fig5_rl_training")
    return figures
