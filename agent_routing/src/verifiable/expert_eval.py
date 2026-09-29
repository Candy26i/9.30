"""Optional dev comparison of prompt-only advisors and frozen role adapters.

This is a role diagnostic, not an independent math benchmark. Teacher labels
measure teacher agreement; only the explicit arithmetic control is executable.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import threading
import time

from .backend import HFBackend
from .expert_train import read_dataset, digest, supervision_metadata
from .protocol import KINDS
from .runner import checkpoint_identity
from .serve import EXPERT_ADAPTERS, FrozenExpertBackend, generate_advisor_request, load_expert_bundle, expert_bundle_sha256
from .telemetry import Monitor, atomic_json, generation, metrics, progress

VERDICTS = ("correct", "incorrect", "uncertain")
ARMS = ("prompt_only", "trained")


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_verdict(text):
    """Read an explicit verdict; conflicting declarations remain unparsed."""
    matches = re.findall(r"(?im)^\s*(?:\*\*|`)?Verdict(?:\*\*|`)?\s*:\s*(?:\*\*|`)?"
                         r"(correct|incorrect|uncertain)(?:\*\*|`)?\s*[.!]?\s*$", text)
    labels = {m.lower() for m in matches}
    return next(iter(labels)) if len(labels) == 1 else None


class BudgetExpired(TimeoutError):
    pass


@contextmanager
def persistent_budget(root, minutes):
    """Accumulate wall time across attempts, charging an unclean exit until resume.

The alarm interrupts an overlong request where Python can handle signals. CUDA
kernel cancellation still depends on the driver; an external job timeout can
provide a stronger process-level limit.
"""
    path = Path(root) / "budget.json"
    budget = json.loads(path.read_text()) if path.exists() else {"minutes": minutes, "spent_seconds": 0.0}
    if budget.get("minutes") != minutes:
        raise ValueError("Expert evaluation budget changed across attempts")
    spent = budget.get("spent_seconds")
    if type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0:
        raise ValueError("Invalid persisted expert evaluation budget")
    if budget.get("active_started_unix") is not None:
        started = budget["active_started_unix"]
        if type(started) not in (int, float) or not math.isfinite(started):
            raise ValueError("Invalid persisted expert evaluation attempt time")
        spent += max(0.0, time.time() - started)
    remaining = minutes * 60 - spent
    if remaining <= 0:
        atomic_json(path, {"minutes": minutes, "spent_seconds": spent, "active_started_unix": None})
        raise BudgetExpired("Persistent expert evaluation wall-time budget exhausted")
    alarm = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    old_handler = None
    if alarm and signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("Expert evaluation cannot replace an existing process alarm")
    start = time.monotonic()
    atomic_json(path, {"minutes": minutes, "spent_seconds": spent, "active_started_unix": time.time()})
    if alarm:
        old_handler = signal.getsignal(signal.SIGALRM)
        def expire(*_):
            raise BudgetExpired("Persistent expert evaluation wall-time budget exhausted")
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining)
    def check():
        if time.monotonic() - start >= remaining:
            raise BudgetExpired("Persistent expert evaluation wall-time budget exhausted")
    try:
        yield check
    finally:
        if alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        atomic_json(path, {"minutes": minutes, "spent_seconds": spent + time.monotonic() - start,
                           "active_started_unix": None})


def _task(row, arm, index):
    return {"arm": arm, "role": row["role"], "row_index": index, "row_sha256": _hash(row),
            "question_hash": row["question_hash"]}


def _shard(root, task):
    return Path(root) / "shards" / task["arm"] / task["role"] / f"{task['row_index']:06d}-{task['row_sha256']}.json"


def _expected_identity(signature, arm, role):
    if arm == "prompt_only":
        return {"role": role, "adapter_name": None, "mode": arm, "identity": signature["base_identity"]}
    return {"role": role, "adapter_name": EXPERT_ADAPTERS[role],
            "identity": signature["bundle"]["roles"][role]["identity"]}


def _validate_result(value, task, max_tokens, expected_identity):
    if (not isinstance(value, dict) or not isinstance(value.get("text"), str)
            or type(value.get("truncated")) is not bool or value.get("actual_role") != task["role"]
            or value.get("advisor_role") != expected_identity
            or any(type(value.get(k)) is not int or value[k] < 0 for k in ("prompt_tokens", "completion_tokens"))
            or value["completion_tokens"] > max_tokens):
        raise ValueError(f"Incomplete expert evaluation result or adapter identity: {task}")


def _read_result(path, task, signature_hash, max_tokens, expected_identity):
    item = json.loads(path.read_text())
    if not isinstance(item, dict) or item.get("task") != task or item.get("run_sha256") != signature_hash:
        raise ValueError(f"Expert evaluation shard identity mismatch: {path}")
    value = item.get("result")
    _validate_result(value, task, max_tokens, expected_identity)
    return value


def _label_basis(row):
    if row.get("label_source") == "teacher_synthetic":
        return "teacher_label_agreement"
    checks = row.get("quality_checks", {})
    if checks.get("scope") == "local_arithmetic_step" and checks.get("arithmetic_verified") is True:
        return "local_arithmetic_verification"
    return "reference_label_agreement"


def _verdict_stats(completed):
    n = len(completed)
    labels = [label for label in VERDICTS if any(row.get("verdict") == label for row, _ in completed)]
    confusion = {truth: {pred: 0 for pred in (*VERDICTS, "unparsed")} for truth in labels}
    for row, output in completed:
        truth = row.get("verdict")
        if truth not in VERDICTS:
            raise ValueError("Verifier development row has no supported reference verdict")
        confusion[truth][parse_verdict(output["text"]) or "unparsed"] += 1
    f1 = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(counts[label] for truth, counts in confusion.items() if truth != label)
        fn = sum(confusion[label].values()) - tp
        f1[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"n": n, "labeled_classes": labels, "confusion": confusion, "class_f1": f1,
            "macro_f1": sum(f1.values()) / len(f1) if f1 else None,
            "accuracy": sum(confusion[label][label] for label in labels) / n if n else None}


def build_report(selected, results):
    bases = sorted({_label_basis(row) for row in selected.get("verifier", [])})
    scopes = {"teacher_label_agreement": "teacher_label_agreement_not_independent_correctness",
              "local_arithmetic_verification": "local_arithmetic_step_only_not_full_proof_verification",
              "reference_label_agreement": "reference_label_agreement_not_independent_correctness"}
    report = {"evaluation_kind": "expert_dev_diagnostic", "independent_benchmark": False,
              "verifier_scope": scopes[bases[0]] if len(bases) == 1 else "mixed_or_empty_label_sources",
              "verifier_label_bases": bases,
              "manual_review_required": ["extractor", "reasoner", "verifier_error_localization"],
              "verdict_scoring": "explicit verdict only, including truncated outputs; incompleteness reported separately",
              "arms": {arm: {} for arm in ARMS}}
    expected = sum(len(rows) for rows in selected.values()) * len(ARMS)
    report["expected_generations"] = expected
    report["completed_generations"] = len(results)
    report["evaluation_complete"] = len(results) == expected
    for arm in ARMS:
        for role, rows in selected.items():
            completed = [(row, results[(arm, role, index)]) for index, row in enumerate(rows)
                         if (arm, role, index) in results]
            n = len(completed)
            stats = {"n_requested": len(rows), "n_completed": n,
                     "unique_questions_requested": len({r["question_hash"] for r in rows}),
                     "unique_questions_completed": len({r["question_hash"] for r, _ in completed}),
                     "empty_count": sum(not r["text"].strip() for _, r in completed),
                     "truncated_count": sum(r["truncated"] for _, r in completed),
                     "context_budget_error_count": sum(r.get("error") == "context_budget_exceeded" for _, r in completed),
                     "quality_assessed": role == "verifier" and bases == ["local_arithmetic_verification"],
                     "quality_scope": "human_review_required"}
            if role == "verifier":
                parsed = sum(parse_verdict(output["text"]) is not None for _, output in completed)
                stats.update(parsed_count=parsed, unparsed_count=n-parsed, parse_rate=parsed / n if n else None,
                             quality_scope=report["verifier_scope"])
                # Separate pseudo-label agreement from executable control scores,
                # including mixed datasets. A common accuracy would hide provenance.
                for basis in bases:
                    measured = _verdict_stats([(row, value) for row, value in completed if _label_basis(row) == basis])
                    stats[basis] = measured
                    if bases == ["local_arithmetic_verification"]:
                        stats.update({k: v for k, v in measured.items() if k not in ("accuracy", "n")})
                        stats["verdict_accuracy"] = measured["accuracy"]
            report["arms"][arm][role] = stats
    if any(basis != "local_arithmetic_verification" for basis in bases):
        report["manual_review_required"].append("verifier_correctness")
    return report


def write_review(root, selected, results, signature_hash):
    records = []
    for role, rows in selected.items():
        for index, row in enumerate(rows):
            outputs = {arm: results[(arm, role, index)] for arm in ARMS if (arm, role, index) in results}
            record = {"role": role, "row_index": index, "question_hash": row["question_hash"],
                      "question": row["question"], "context": row.get("context", ""),
                      "candidate": row.get("candidate", ""), "prompt": row["prompt"],
                      "reference_response": row["response"], "reference_is_model_input": False,
                      "outputs": outputs, "run_sha256": signature_hash,
                      "human_review": {"status": "pending", "quality": None, "error_localization": None}}
            for key in ("source_id", "source_revision", "source_record_sha256", "reference_sha256",
                        "label_source", "quality_checks", "verdict", "arithmetic_evidence", "corruption", "reviewed",
                        "teacher_provider", "teacher_model", "teacher_revision", "teacher_actual_model",
                        "teacher_request_id", "teacher_response_sha256", "teacher_prompt_sha256",
                        "candidate_request_id", "candidate_response_sha256", "candidate_terminal_diagnostic"):
                if key in row:
                    record[key] = row[key]
            if role == "verifier":
                record["verdict_label_basis"] = _label_basis(row)
                record["parsed_verdicts"] = {arm: parse_verdict(value["text"]) for arm, value in outputs.items()}
            records.append(record)
    path = Path(root) / "review.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    temporary.replace(path)


def evaluate(bundle_file, data_dir, output, limit=32, max_tokens=512, minutes=120, max_context=16384):
    if any(type(v) is not int or v <= 0 for v in (limit, max_tokens, max_context)):
        raise ValueError("limit, max_tokens and max_context must be positive integers")
    if type(minutes) not in (int, float) or not math.isfinite(minutes) or not 0 < minutes <= 120:
        raise ValueError("minutes must be in (0, 120]")
    from .provenance import harness_identity
    bundle = load_expert_bundle(bundle_file)
    selected = {}
    for role in KINDS:
        data_manifest, _, rows = read_dataset(data_dir, role)
        selected[role] = rows[:limit]
        if role == "verifier" and any(row.get("verdict") not in VERDICTS for row in selected[role]):
            raise ValueError("Verifier development labels must be correct, incorrect or uncertain")
    signature = {"schema_version": 1, "stage": "expert_eval", "bundle": bundle,
                 "expert_bundle_sha256": expert_bundle_sha256(bundle),
                 "base_identity": checkpoint_identity(bundle["base_model"]), "harness": harness_identity(),
                 "data_manifest_sha256": digest(Path(data_dir) / "manifest.json"),
                 "data_supervision": supervision_metadata(data_manifest),
                 "selected_rows": {role: [_hash(row) for row in rows] for role, rows in selected.items()},
                 "config": {"limit": limit, "max_tokens": max_tokens, "minutes": minutes,
                            "max_context": max_context, "temperature": 0.0, "seed": 42}}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another expert evaluation is using this output directory") from exc
        run_path = root / "run.json"
        if run_path.exists():
            if json.loads(run_path.read_text()) != signature:
                raise ValueError("Expert evaluation configuration/data/bundle changed; use a new output directory")
        elif any(path.name != ".eval.lock" for path in root.iterdir()):
            raise ValueError("Expert evaluation output is not empty and has no matching manifest")
        else:
            atomic_json(run_path, signature)
        signature_hash = _hash(signature)
        results = {}
        for arm in ARMS:
            for role, rows in selected.items():
                for index, row in enumerate(rows):
                    task = _task(row, arm, index)
                    path = _shard(root, task)
                    if path.exists():
                        results[(arm, role, index)] = _read_result(path, task, signature_hash, max_tokens,
                                                                 _expected_identity(signature, arm, role))
        report = build_report(selected, results)
        if report["evaluation_complete"]:
            write_review(root, selected, results, signature_hash)
            report.update(controller_status="completed", run_sha256=signature_hash, provenance=signature)
            if (root / "budget.json").exists():
                report["budget"] = json.loads((root / "budget.json").read_text())
            atomic_json(root / "report.json", report)
            return report
        with Monitor(root, "expert_eval") as monitor:
            monitor.summary({"independent_benchmark": False, "evaluation_complete": False,
                             "expert_bundle_sha256": signature["expert_bundle_sha256"]})
            status, failure = "completed", None
            try:
                with persistent_budget(root, minutes) as check_budget:
                    for arm in ARMS:
                        pending = [(role, index, row) for role, rows in selected.items() for index, row in enumerate(rows)
                                   if (arm, role, index) not in results]
                        if not pending:
                            continue
                        check_budget()
                        progress(phase="loading_model", arm=arm)
                        backend = (HFBackend(bundle["base_model"], max_context=max_context,
                                   revision=bundle["base_model_revision"], usage_actor="advisor") if arm == "prompt_only"
                                   else FrozenExpertBackend(bundle, max_context))
                        try:
                            for role, index, row in pending:
                                check_budget()
                                progress(phase="expert_dev", arm=arm, expert_role=role, row_index=index,
                                         completed_generations=len(results), expected_generations=report["expected_generations"])
                                monitor.question = {"question_hash": row["question_hash"], "question": row["question"],
                                                    "context": row.get("context", ""), "split": "dev"}
                                value, _ = generate_advisor_request(backend, {"model": role, "messages": row["prompt"],
                                    "max_tokens": max_tokens, "temperature": 0.0, "seed": 42})
                                if arm == "prompt_only":
                                    value.update(actual_role=role, advisor_role=_expected_identity(signature, arm, role))
                                task = _task(row, arm, index)
                                _validate_result(value, task, max_tokens, _expected_identity(signature, arm, role))
                                atomic_json(_shard(root, task), {"task": task, "run_sha256": signature_hash, "result": value})
                                results[(arm, role, index)] = value
                                generation("advisor", value, messages=row["prompt"], advisor=role, operation=arm, max_tokens=max_tokens)
                                report = build_report(selected, results)
                                monitor.summary(report)
                                metrics(report["arms"], "expert_dev", diagnostic_step=len(results))
                        finally:
                            del backend
                            gc.collect()
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
            except BudgetExpired as exc:
                status, failure = "budget_exhausted", str(exc)
            except BaseException as exc:
                status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                failure = str(exc)
                raise
            finally:
                report = build_report(selected, results)
                report.update(controller_status=status, error=failure, run_sha256=signature_hash, provenance=signature)
                if (root / "budget.json").exists():
                    report["budget"] = json.loads((root / "budget.json").read_text())
                write_review(root, selected, results, signature_hash)
                atomic_json(root / "report.json", report)
                monitor.summary(report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True, help="Matching interrupted outputs resume from committed generation shards")
    parser.add_argument("--limit", type=int, default=32, help="Maximum dev rows per role, not unique questions")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--minutes", type=float, default=120)
    args = parser.parse_args()
    report = evaluate(args.bundle, args.data_dir, args.out, args.limit, args.max_tokens, args.minutes, args.max_context)
    print(json.dumps({key: report[key] for key in ("controller_status", "evaluation_complete", "completed_generations", "expected_generations")}))
    if not report["evaluation_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
