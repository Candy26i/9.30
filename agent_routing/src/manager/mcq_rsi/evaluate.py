"""MCQ RSI evaluation: the paper's eval harness with cached fail-stop advisors, metrics and gates.

``evaluate`` runs ``stages.run_eval_manager_tools`` exactly as the paper did
(greedy, ``max_new_tokens`` 1024, ``max_tool_calls`` 3, deployment tool schema
``_manager_tool_schemas``, the benchmark's task description) with three changes
around it, none inside the loop:

- ``binding_mode="environment"`` is forced (``auto`` falls back to ``argument``
  without ``manager_run_config.json``, ``stages._resolve_binding_mode``);
- advisors come from the ``CachedAdvisorPool`` injected through the new ``pool``
  parameter, wrapped in ``FailStopEvalPool``: an advisor failure aborts the eval
  (``AdvisorFailStop``) instead of reaching the manager as ``{"error": ...}``;
  ``clear_abort`` is called first so a pool aborted by an earlier stage is reusable;
- outputs go to the stage's own ``out_dir`` (new ``out_dir`` parameter).

Two-pass speculative Verifier prefetch: pass 1 generates every question's first
manager turn with the same loader, render and decoding as the eval (so it is the
eval's own first turn) and prefetches, concurrently, the advisors that turn calls
(the Verifier with its ``current_draft``) and, when it calls E or R, the
speculative V(q, X) on its draft X, the usual second call. Pass 2 is the eval; a
call pass 1 did not predict (a Verifier call on a revised draft, a second E/R
call not already cached by the ``prefetch/advisors`` stage) is fetched in line.

``evaluate_forced`` wraps ``stages.run_eval_manager_forced`` the same way (forced
Verifier without a candidate, as in the paper). ``eval_metrics`` recomputes the
§5 metrics from the per-example records and checks them against the stages
report; ``eval_gate``, ``grpo_accept`` and ``sft_flags`` are the §3.6 gates.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...benchmarks.base import StandardRow
from ..marginal_value import ADVISOR_KINDS
from . import advisors as advisor_mod
from . import benchmarks as registry
from . import protocol
from .advisors import AdvisorRequest, FailStopEvalPool
from .collect import _atomic_write, rows_digest

EVAL_VERSION = "mcq_rsi_eval/1"
BINDING = "environment"
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 1024
MAX_TOOL_CALLS = 3
STAGE_SEED = 42  # StageContext default: the paper eval's sample shuffle (n_samples = all rows)
ACC_TOLERANCE = 0.01  # GRPO accepted if dev accuracy >= S_k - 1.0 pt
EXTRA_CALLS = 0.15  # ... calls/example <= S_k + 0.15
GAP_FRACTION = 0.5  # ... call gap >= S_k's - 0.5 * |S_k's| (= 0.5 * S_k's for a non-negative S_k gap)
SFT_MAX_CALLS = 1.5
SFT_MAX_ACC_DROP = 0.03
_EPS = 1e-12


class EvalGateFailed(RuntimeError):
    def __init__(self, message: str, result: Dict[str, Any]):
        super().__init__(message)
        self.result = result


# --------------------------------------------------------------------------- plumbing

def stage_context(base_model: str, out_dir, seed: int = STAGE_SEED):
    """A ``StageContext`` whose derived directories stay inside the stage directory."""
    from ...pipeline.stages import StageContext
    return StageContext(base_model=base_model, teacher_id="mcq_rsi_eval",
                        output_root=str(Path(out_dir) / ".stage_ctx"), seed=seed, binding_mode=BINDING)


def standard_rows(rows: Sequence[Dict[str, Any]]) -> List[StandardRow]:
    out = []
    for r in rows:
        if r.get("ground_truth") not in r["choices"]:
            raise ValueError(f"example {r.get('example_id')}: ground truth is not a choice key")
        out.append(StandardRow(example_id=int(r["example_id"]), benchmark_name=str(r.get("benchmark_name") or ""),
                               task_subtype=str(r.get("task_subtype") or ""), question=r["question"],
                               choices=dict(r["choices"]), ground_truth=r["ground_truth"],
                               context=r.get("context") or "", metadata=dict(r.get("metadata") or {}),
                               split=str(r.get("split") or "")))
    eids = [r.example_id for r in out]
    if len(set(eids)) != len(eids):
        raise ValueError("duplicate example_id in the eval rows")
    return out


def fail_stop(pool) -> FailStopEvalPool:
    """``FailStopEvalPool`` around ``pool`` (idempotent), with any earlier abort cleared."""
    wrapped = pool if isinstance(pool, FailStopEvalPool) else FailStopEvalPool(pool)
    if hasattr(wrapped, "clear_abort"):
        wrapped.clear_abort()
    return wrapped


class _Stats:
    """Advisor counters of one eval (``eval_gate`` reads ``.stats``)."""

    def __init__(self, stats: Dict[str, int]):
        self.stats = stats


def _write_json(path: Path, value) -> None:
    _atomic_write(Path(path), json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def base_identity(base_model: str, revision: Optional[str] = registry.BASE_REVISION) -> Dict[str, Any]:
    """What ``base_model`` resolves to when stages / ``train_manager_sft`` load it by name (no revision).

    A local directory: its files (``checkpoint_identity``) and, for an HF snapshot
    directory, the commit. A hub id: the commit the local HF cache resolves ``main``
    to (no network; None when not cached). FA-GRPO loads ``registry.BASE_REVISION``,
    so for the registry base a different commit is refused: the eval or SFT would run
    on another base than the KL reference. The CLI passes a local snapshot of
    ``BASE_MODEL@BASE_REVISION`` (``resolve_base``), which pins it.
    """
    from ...verifiable.runner import checkpoint_identity
    repo = "models--" + registry.BASE_MODEL.replace("/", "--")
    path = Path(base_model)
    if path.is_dir():
        commit = path.name if path.parent.name == "snapshots" else None
        if revision and commit is not None and path.parent.parent.name == repo and commit != revision:
            raise ValueError(f"{base_model} is {registry.BASE_MODEL}@{commit}, not the pinned revision {revision}")
        return {"path": str(path.resolve()), "commit": commit, "files": checkpoint_identity(path)}
    commit = None
    try:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(base_model, "config.json")
        if isinstance(cached, str):
            commit = Path(cached).parent.name
    except Exception:  # noqa: BLE001 -- identity only; an unusable cache leaves the commit unknown
        commit = None
    if base_model == registry.BASE_MODEL and revision and commit is not None and commit != revision:
        raise ValueError(f"{base_model} resolves to commit {commit} in the HF cache, not the pinned revision "
                         f"{revision} FA-GRPO uses; pass a local snapshot directory of {revision} as the base model")
    return {"model_id": base_model, "commit": commit}


def resolve_base(base_model: str, revision: Optional[str] = registry.BASE_REVISION) -> str:
    """A local snapshot directory of ``base_model@revision`` (a local directory is returned unchanged)."""
    if os.path.isdir(base_model) or not revision:
        return base_model
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(base_model, revision=revision, local_files_only=True)
    except Exception:  # noqa: BLE001 -- not cached yet: download it
        return snapshot_download(base_model, revision=revision)


def code_identity() -> Dict[str, Any]:
    """Code and packages that shape an eval beyond its inputs (stages loop, prompts, this package)."""
    from ...verifiable.provenance import harness_identity
    from .collect import source_sha256
    return {"harness": harness_identity(), "source_sha256": source_sha256()}


def _signature(kind: str, checkpoint, rows, pool, base_model: str, settings: Dict[str, Any],
               base_revision: Optional[str] = registry.BASE_REVISION) -> Dict[str, Any]:
    from ...verifiable.runner import checkpoint_identity
    return json.loads(json.dumps({
        "version": EVAL_VERSION, "kind": kind, "checkpoint": str(checkpoint),
        "checkpoint_identity": checkpoint_identity(checkpoint), "base_model": base_model,
        "base_revision": base_revision, "base_identity": base_identity(base_model, base_revision), **code_identity(),
        "rows_sha256": rows_digest(rows), "n_rows": len(rows),
        "advisors": pool.identity() if hasattr(pool, "identity") else None, "settings": settings,
    }))


def _resume(result_path: Path, signature: Dict[str, Any], require_gate: bool) -> Optional[Dict[str, Any]]:
    if not result_path.exists():
        return None
    old = json.loads(result_path.read_text(encoding="utf-8"))
    if old.get("signature") != signature:
        raise ValueError(f"{result_path} holds an eval of different inputs; use a new stage directory")
    if require_gate and old["gate"]:
        raise EvalGateFailed(f"eval gate failed: {old['gate']}", old)
    return old


# ----------------------------------------------------------------- speculative prefetch

def first_turns(ctx, checkpoint, rows: Sequence[StandardRow], bench: registry.Benchmark,
                max_new_tokens: int = MAX_NEW_TOKENS) -> List[Dict[str, Any]]:
    """Pass 1: every question's first manager turn, generated exactly as ``run_eval_manager_tools`` does."""
    import torch

    from ...pipeline import stages
    from ..prompt import parse_draft_answer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok, model = stages._load_manager_for_eval(ctx, str(checkpoint), device, dtype)
    tools = stages._manager_tool_schemas(BINDING)
    out = []
    try:
        for r in rows:
            messages = protocol.manager_messages(bench, r.to_dict())
            inputs = tok(stages._render_manager_chat(tok, messages, tools), return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            text = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            content, calls = stages._extract_manager_tool_calls(text)
            out.append({"example_id": r.example_id, "draft": parse_draft_answer(content or text, list(r.choices)),
                        "calls": calls})
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def speculative_requests(rows: Sequence[StandardRow], turns: Sequence[Dict[str, Any]]) -> List[AdvisorRequest]:
    """The first turn's own calls (exact, Verifier with its ``current_draft``) plus, when that turn calls
    another advisor, the speculative V(q, X) on its draft X. A committing root never calls."""
    by_id = {r.example_id: r for r in rows}
    requests = []
    for turn in turns:
        if not turn["calls"]:
            continue
        row = by_id[turn["example_id"]].to_dict()
        kinds = [c["name"][:-5] for c in turn["calls"] if c["name"][:-5] in ADVISOR_KINDS]
        candidates = [str(c["arguments"].get("current_draft") or "") for c in turn["calls"]
                      if c["name"] == "verifier_tool"]
        if not candidates and turn["draft"]:
            candidates = [turn["draft"]]
        requests += [AdvisorRequest.for_row(k, row) for k in dict.fromkeys(kinds) if k != "verifier"]
        requests += [AdvisorRequest.for_row("verifier", row, c) for c in dict.fromkeys(candidates)]
    return requests


# ------------------------------------------------------------------------- metrics

DUPLICATE_CALL = "tool_already_called"
# stages.run_eval_manager_tools' own reply, byte for byte, when the manager calls an advisor kind it already called.
DUPLICATE_PAYLOAD = '{"error": "tool_already_called", "detail": "each tool may be used at most once"}'


def _error_payload(text: str) -> Optional[str]:
    """The ``error`` value of a JSON-dict tool output, else None."""
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return None
    return str(obj["error"]) if isinstance(obj, dict) and "error" in obj else None


def tool_output_events(record: Dict[str, Any]) -> Dict[str, int]:
    """``{"advisor_errors", "duplicate_calls"}`` of one eval record's tool outputs.

    A duplicate call is a manager protocol event: stages' exact ``DUPLICATE_PAYLOAD`` answering a tool
    whose kind already received an output earlier in the same record. Kinds follow stages
    (``verifier`` and ``verifier_tool`` are one kind). Every other ``{"error": ...}`` output
    (including that text from a first call of an advisor) is an advisor error.
    """
    seen, out = set(), {"advisor_errors": 0, "duplicate_calls": 0}
    for e in record.get("trajectory") or []:
        if e.get("role") != "tool":
            continue
        content, name = e.get("content"), str(e.get("name") or "")
        kind = name[:-5] if name.endswith("_tool") else name
        if content == DUPLICATE_PAYLOAD and kind in seen:
            out["duplicate_calls"] += 1
        elif _error_payload(content) is not None:
            out["advisor_errors"] += 1
        seen.add(kind)
    return out


def eval_metrics(records: Sequence[Dict[str, Any]], report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """§5 metrics from ``manager_tool_eval.jsonl`` records (cross-checked with the stages report)."""
    n = len(records)
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    with_draft = [r for r in records if r.get("initial_draft") is not None]
    wrong = [r for r in with_draft if not r["initial_draft_correct"]]
    right = [r for r in with_draft if r["initial_draft_correct"]]
    advisors: Dict[str, int] = {}
    first: Dict[str, int] = {}
    for r in records:
        for name in r.get("tool_names_called") or []:
            advisors[name] = advisors.get(name, 0) + 1
        if r.get("tool_names_called"):
            first[r["tool_names_called"][0]] = first.get(r["tool_names_called"][0], 0) + 1
    events = [tool_output_events(r) for r in records]
    accuracy = mean([bool(r["correct"]) for r in records])
    draft_accuracy = mean([bool(r["initial_draft_correct"]) for r in with_draft])
    p_wrong, p_right = mean([r["tool_calls"] > 0 for r in wrong]), mean([r["tool_calls"] > 0 for r in right])
    metrics = {
        "n": n,
        "accuracy": accuracy,
        "initial_draft_accuracy": draft_accuracy,
        "initial_draft_coverage": len(with_draft) / n if n else 0.0,
        "gain_pp": 100.0 * (accuracy - draft_accuracy),
        "calls_per_example": mean([r["tool_calls"] for r in records]),
        "call_rate": mean([r["tool_calls"] > 0 for r in records]),
        "per_advisor_calls": dict(sorted(advisors.items())),
        "first_call_counts": dict(sorted(first.items())),
        "correction_rate": mean([bool(r["corrected_by_tools"]) for r in records]),  # unconditional (Table 8)
        "corruption_rate": mean([bool(r["corrupted_by_tools"]) for r in records]),
        "call_rate_given_draft_wrong": p_wrong,
        "call_rate_given_draft_correct": p_right,
        "call_gap": p_wrong - p_right,
        "valid_answer_rate": mean([bool(r["valid_answer"]) for r in records]),
        "error_payload_tool_outputs": sum(e["advisor_errors"] for e in events),  # gated: advisor error reached the manager
        "duplicate_tool_calls": sum(e["duplicate_calls"] for e in events),  # not gated: manager repeated a tool
        "malformed_tool_calls": None if report is None else report.get("malformed_tool_calls"),
    }
    if report is not None:
        pairs = {"accuracy": "accuracy", "initial_draft_accuracy": "initial_draft_accuracy",
                 "initial_draft_coverage": "initial_draft_coverage", "calls_per_example": "avg_tool_calls",
                 "call_rate": "tool_call_rate", "correction_rate": "correction_rate",
                 "corruption_rate": "corruption_rate", "call_gap": "draft_conditioned_call_gap",
                 "valid_answer_rate": "valid_answer_rate"}
        for ours, theirs in pairs.items():
            if abs(metrics[ours] - report[theirs]) > 1e-9:
                raise ValueError(f"metric {ours}={metrics[ours]} disagrees with the stages report {theirs}={report[theirs]}")
        if metrics["per_advisor_calls"] != dict(sorted(report.get("tool_counts", {}).items())) or report["n_samples"] != n:
            raise ValueError("per-advisor counts / n disagree with the stages report")
    return metrics


def forced_metrics(records: Sequence[Dict[str, Any]], report: Dict[str, Any]) -> Dict[str, Any]:
    n = len(records)
    out = {"n": n, "accuracy": sum(bool(r["correct"]) for r in records) / max(1, n),
           "valid_answer_rate": sum(bool(r["valid_answer"]) for r in records) / max(1, n),
           "calls_per_example": float(len(report["forced_tools"])), "forced_tools": list(report["forced_tools"])}
    if abs(out["accuracy"] - report["accuracy"]) > 1e-9 or report["n_samples"] != n:
        raise ValueError("forced metrics disagree with the stages report")
    return out


# ---------------------------------------------------------------------------- gates

def eval_gate(report: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None, pool=None) -> List[str]:
    """§3.6, every eval: no advisor error reached the manager and every answer parsed ([] = pass).

    stages' own ``tool_already_called`` reply to a repeated advisor kind is manager
    behaviour, not an advisor error: it is reported as ``duplicate_tool_calls`` and not gated.
    """
    if "forced_tools" in report:
        failures = [] if report.get("valid_answer_rate") == 1.0 else [f"valid_answer_rate={report.get('valid_answer_rate')}"]
        if pool is not None and getattr(pool, "stats", {}).get("failed", 0):
            failures.append(f"advisor failures={pool.stats['failed']}")
    else:
        failures = advisor_mod.eval_gate(report, pool)
    if metrics and metrics.get("error_payload_tool_outputs"):
        failures.append(f"error payloads in tool outputs={metrics['error_payload_tool_outputs']}")
    return failures


def _metrics(x: Dict[str, Any]) -> Dict[str, Any]:
    return x.get("metrics", x)


def grpo_accept(grpo: Dict[str, Any], sft: Dict[str, Any], informative: bool, *,
                acc_tolerance: float = ACC_TOLERANCE, extra_calls: float = EXTRA_CALLS,
                gap_fraction: float = GAP_FRACTION) -> Dict[str, Any]:
    """§3.6: accept G_k iff dev acc >= S_k - 1 pt, calls <= S_k + 0.15, call gap >= 0.5 * S_k's,
    and FA-GRPO was informative; otherwise ``G_k := S_k`` (``grpo_rejected``).

    The gap bound is ``s - (1 - gap_fraction) * |s|``: exactly ``0.5 * s`` for s >= 0, and for a
    negative S_k gap (``0.5 * s > s``) a G_k at least as good as S_k is never rejected.

    ``grpo`` / ``sft`` are ``evaluate`` results (their eval gates must pass) or metric dicts.
    """
    g, s = _metrics(grpo), _metrics(sft)
    reasons = []
    for name, result in (("G_k", grpo), ("S_k", sft)):
        if result.get("gate"):
            reasons.append(f"{name} dev eval failed its gate: {result['gate']}")
    if g["accuracy"] < s["accuracy"] - acc_tolerance - _EPS:
        reasons.append(f"accuracy {g['accuracy']:.4f} < S_k {s['accuracy']:.4f} - {acc_tolerance}")
    if g["calls_per_example"] > s["calls_per_example"] + extra_calls + _EPS:
        reasons.append(f"calls {g['calls_per_example']:.4f} > S_k {s['calls_per_example']:.4f} + {extra_calls}")
    gap_bound = s["call_gap"] - (1 - gap_fraction) * abs(s["call_gap"])
    if g["call_gap"] < gap_bound - _EPS:
        reasons.append(f"call gap {g['call_gap']:.4f} < {gap_bound:.4f} (S_k {s['call_gap']:.4f}, fraction {gap_fraction})")
    if not informative:
        reasons.append("FA-GRPO stage was uninformative (< 15% revision states with J in (0.02, 0.98))")
    return {"accepted": not reasons, "reasons": reasons, "decision": "grpo_accepted" if not reasons else "grpo_rejected",
            "deltas": {"accuracy": g["accuracy"] - s["accuracy"],
                       "calls_per_example": g["calls_per_example"] - s["calls_per_example"],
                       "call_gap": g["call_gap"] - s["call_gap"]}}


def sft_flags(sft: Dict[str, Any], previous: Optional[Dict[str, Any]] = None, *,
              max_calls: float = SFT_MAX_CALLS, max_acc_drop: float = SFT_MAX_ACC_DROP) -> List[str]:
    """§3.6: SFT is never selected on dev, only flagged (calls > 1.5, or accuracy down > 3 pts)."""
    s = _metrics(sft)
    flags = []
    if s["calls_per_example"] > max_calls + _EPS:
        flags.append(f"calls/example {s['calls_per_example']:.3f} > {max_calls}")
    if previous is not None and s["accuracy"] < _metrics(previous)["accuracy"] - max_acc_drop - _EPS:
        flags.append(f"accuracy {s['accuracy']:.4f} dropped > {max_acc_drop} from {_metrics(previous)['accuracy']:.4f}")
    return flags


# ---------------------------------------------------------------------------- stages

def evaluate(checkpoint, rows: Sequence[Dict[str, Any]], pool, out_dir, *, bench: str,
             base_model: str = registry.BASE_MODEL, base_revision: Optional[str] = registry.BASE_REVISION,
             speculative: bool = True, max_new_tokens: int = MAX_NEW_TOKENS, max_tool_calls: int = MAX_TOOL_CALLS,
             seed: int = STAGE_SEED, require_gate: bool = True) -> Dict[str, Any]:
    """Free-routing dev/test eval of one checkpoint (paper protocol); writes ``mcq_rsi_eval.json``.

    The base is pinned: a hub id is replaced by its local snapshot at ``base_revision``
    (``resolve_base``), which is what stages loads; the revision and the resolved base's
    identity are in the signature. ``base_revision`` None / "" leaves a hub id unpinned.
    """
    from ...pipeline import stages
    spec = registry.get(bench)
    base_revision = base_revision or None
    base_model = resolve_base(base_model, base_revision)
    out = Path(out_dir)
    settings = {"mode": "tools", "binding_mode": BINDING, "temperature": TEMPERATURE, "max_new_tokens": max_new_tokens,
                "max_tool_calls": max_tool_calls, "seed": seed, "task_description": spec.task_description,
                "tool_schema": "stages._manager_tool_schemas(environment)"}
    signature = _signature("tools", checkpoint, rows, pool, base_model, settings, base_revision)
    result_path = out / "mcq_rsi_eval.json"
    done = _resume(result_path, signature, require_gate)
    if done is not None:
        return done
    srows = standard_rows(rows)
    guarded = fail_stop(pool)
    ctx = stage_context(base_model, out, seed)
    stats_before = dict(getattr(guarded, "stats", {}) or {})
    prefetch = None
    if speculative and hasattr(guarded, "prefetch"):
        turns = first_turns(ctx, checkpoint, srows, spec, max_new_tokens)
        requests = speculative_requests(srows, turns)
        try:
            prefetch = {**guarded.prefetch(requests), "first_turn_calls": sum(bool(t["calls"]) for t in turns)}
        except advisor_mod.AdvisorError as e:
            guarded.abort()
            raise advisor_mod.AdvisorFailStop(str(e)) from e
    report = stages.run_eval_manager_tools(
        ctx, srows, manager_dir=str(checkpoint), n_samples=len(srows), temperature=TEMPERATURE,
        max_new_tokens=max_new_tokens, max_tool_calls=max_tool_calls, task_description=spec.task_description,
        subagent_server_url=None, pool=guarded, out_dir=str(out))
    records = _read_jsonl(out / "manager_tool_eval.jsonl")
    metrics = eval_metrics(records, report)
    stats = {k: v - stats_before.get(k, 0) for k, v in (getattr(guarded, "stats", {}) or {}).items()}
    gate = eval_gate(report, metrics, _Stats(stats))  # failures of this eval only, not of earlier stages
    result = {"version": EVAL_VERSION, "signature": signature, "metrics": metrics, "gate": gate,
              "passed": not gate, "prefetch": prefetch, "report": str(out / "manager_tool_eval_report.json"),
              "records": str(out / "manager_tool_eval.jsonl"), "advisor_stats": stats}
    _write_json(result_path, result)
    if require_gate and gate:
        raise EvalGateFailed(f"eval gate failed: {gate}", result)
    return result


def evaluate_forced(checkpoint, rows: Sequence[Dict[str, Any]], pool, out_dir, forced_tools: Sequence[str], *,
                    bench: str, base_model: str = registry.BASE_MODEL,
                    base_revision: Optional[str] = registry.BASE_REVISION, max_new_tokens: int = MAX_NEW_TOKENS,
                    seed: int = STAGE_SEED, require_gate: bool = True) -> Dict[str, Any]:
    """Forced-delegation baseline (forced-V / forced-all, Tables 4-5); Verifier gets no candidate.

    The base is pinned as in ``evaluate``."""
    from ...pipeline import stages
    spec = registry.get(bench)
    base_revision = base_revision or None
    base_model = resolve_base(base_model, base_revision)
    forced = [t for t in forced_tools if t and t != "none"]
    unknown = sorted(set(forced) - set(ADVISOR_KINDS))
    if unknown:
        raise ValueError(f"unknown forced tools {unknown}")
    tag = ",".join(forced) or "none"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    out = Path(out_dir)
    settings = {"mode": "forced", "forced_tools": forced, "binding_mode": BINDING, "temperature": TEMPERATURE,
                "max_new_tokens": max_new_tokens, "seed": seed, "task_description": spec.task_description}
    signature = _signature("forced", checkpoint, rows, pool, base_model, settings, base_revision)
    result_path = out / f"mcq_rsi_forced_{safe}.json"
    done = _resume(result_path, signature, require_gate)
    if done is not None:
        return done
    srows = standard_rows(rows)
    guarded = fail_stop(pool)
    stats_before = dict(getattr(guarded, "stats", {}) or {})
    if hasattr(guarded, "prefetch") and forced:
        try:
            guarded.prefetch([AdvisorRequest.for_row(k, r.to_dict()) for r in srows for k in forced])
        except advisor_mod.AdvisorError as e:
            guarded.abort()
            raise advisor_mod.AdvisorFailStop(str(e)) from e
    report = stages.run_eval_manager_forced(
        stage_context(base_model, out, seed), srows, manager_dir=str(checkpoint), forced_tools=forced,
        n_samples=len(srows), temperature=TEMPERATURE, max_new_tokens=max_new_tokens,
        task_description=spec.task_description, out_tag=tag, subagent_server_url=None, pool=guarded, out_dir=str(out))
    records = _read_jsonl(out / f"manager_forced_{safe}.jsonl")
    metrics = forced_metrics(records, report)
    stats = {k: v - stats_before.get(k, 0) for k, v in (getattr(guarded, "stats", {}) or {}).items()}
    gate = eval_gate(report, None, _Stats(stats))
    result = {"version": EVAL_VERSION, "signature": signature, "metrics": metrics, "gate": gate, "passed": not gate,
              "report": str(out / f"manager_forced_{safe}_report.json"), "records": str(out / f"manager_forced_{safe}.jsonl")}
    _write_json(result_path, result)
    if require_gate and gate:
        raise EvalGateFailed(f"eval gate failed: {gate}", result)
    return result

