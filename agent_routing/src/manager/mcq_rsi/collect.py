"""On-policy counterfactual collection for MCQ RSI (design §3.3).

One record per root question, in the paper schema of
``marginal_value.build_marginal_value_sft`` (accepted unchanged by
``summarize_counterfactuals`` and ``_make_sft_rows``) plus ``split``, ``pool``,
``round``, ``root_mode``, ``policy_action`` and ``unconstrained_argmax``.

``root_mode="policy"`` (RSI): the root is the manager's own grammar-constrained
greedy deployment turn (``TOOLS_DEPLOY``, no probe message); its draft key is
the shared root draft and its full action is recorded as ``policy_action``.
Each branch forces ``DRAFT_ANSWER_X`` + call(a) (Verifier gets
``current_draft=X``) and appends the cached tool output; the manager then
revises greedily with the commit forced, ``DRAFT_ANSWER_Y\\nANSWER_Z`` (paper
Eq. 2), and the branch outcome is Z (paper and eval semantics; Y is kept in
``revision.draft``). The revision prompt shows the call as the deployed eval
keeps it in the history (``example_id`` injected); ``trajectory`` stores the
paper call turn, so SFT rows stay byte-compatible with round 1. Branch
revisions of one depth go to the manager together. ``root_mode="probe"``
reproduces the paper's ``_DIRECT_PROBE``/``_REVISION_PROBE`` generation (paper
call turns throughout) for parity tests only.

Search is the paper's breadth-first rule: a correct root expands depth 1 only,
a wrong root stops at the first depth with a success, ties among the shortest
successes are broken by ``Random(seed + example_id)``, unsolved roots get no label.

Every question is written to its own atomic shard; ``resume`` skips finished
shards after checking the run identity (``collect_manifest.json``: rows, seed,
depth, manager checkpoint + adapter sha + chat template + batch size + dtype +
device, advisor cache identity, ``harness_identity`` incl. the fla /
causal-conv1d kernel packages, and ``source_sha256`` of the code that shapes
records) and validating every existing shard (parseable, its file name, example
id, question hash, pool, round and root mode match the run; no shard outside
the pool), so kill + resume gives byte-identical outputs. A directory with
shards but no manifest is refused. Advisor failures raise
(``advisors.AdvisorError``) before the shard of the current question is written.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...benchmarks.base import question_hash
from ...verifiable.provenance import harness_identity
from ..marginal_value import (
    _DIRECT_PROBE,
    _REVISION_PROBE,
    ADVISOR_KINDS,
    choose_preferred_sequence,
    summarize_counterfactuals,
)
from . import protocol
from .advisors import AdvisorRequest

COLLECTOR_VERSION = "mcq_rsi_collect/1"
ROOT_MODES = ("policy", "probe")
TEXT_CAP = 1200  # paper record cap for direct_text / probe_text
DEFAULT_SEED = 42  # paper seed: tie-break Random(seed + example_id)
PACKAGE = Path(__file__).resolve().parent
SOURCES = (PACKAGE, PACKAGE.parent / "prompt.py", PACKAGE.parents[1] / "pipeline/stages.py")


@dataclass
class _State:
    sequence: Tuple[str, ...]
    messages: List[Dict[str, Any]]
    drafts: List[str]
    trajectory: List[Dict[str, Any]]


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def source_sha256() -> str:
    """Code that shapes records beyond ``harness_identity`` (this package, prompts, the eval/schema module)."""
    files = sorted(f for src in SOURCES for f in ([src] if src.is_file() else src.rglob("*"))
                   if f.is_file() and f.suffix in (".py", ".txt", ".jinja"))
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f.relative_to(PACKAGE.parents[1])).encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()


def _check_shards(shards: Path, rows: Sequence[Dict[str, Any]], identity: Dict[str, Any]) -> None:
    """Refuse to resume over shards that are not this run's (never trust a file name alone)."""
    by_name = {f"{int(r['example_id'])}.json": r for r in rows}
    for path in sorted(shards.glob("*.json")) if shards.is_dir() else []:
        row = by_name.get(path.name)
        if row is None:
            raise ValueError(f"{path}: shard for an example outside this pool; refusing to resume")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"{path}: unreadable shard ({e}); refusing to resume") from e
        expected = {"example_id": int(row["example_id"]), "question_hash": question_hash(row["question"]),
                    "pool": identity["pool"], "round": identity["round"], "root_mode": identity["root_mode"],
                    "split": "train"}
        got = {k: record.get(k) for k in expected} if isinstance(record, dict) else {}
        if got != expected or not isinstance(record.get("branches"), list):
            bad = sorted(k for k in expected if got.get(k) != expected[k])
            raise ValueError(f"{path}: shard does not belong to this run (differs in {bad or ['branches']})")


def rows_digest(rows: Sequence[Dict[str, Any]]) -> str:
    items = [[int(r["example_id"]), question_hash(r["question"])] for r in rows]
    return hashlib.sha256(json.dumps(items).encode("utf-8")).hexdigest()


def _advise(pool, row, kind: str, draft: str) -> str:
    return pool.call(agent_kind=kind, example_id=int(row["example_id"]), question=row["question"],
                     context=row.get("context") or "", choices=row["choices"], cache_namespace="marginal_value",
                     candidate_answer=draft if kind == "verifier" else "")


def collect_question(row: Dict[str, Any], bench, manager, pool, *, root_mode: str = "policy",
                     max_depth: int = 2, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """The counterfactual tree of one root question (paper record schema + policy fields)."""
    if root_mode not in ROOT_MODES:
        raise ValueError(f"root_mode must be one of {ROOT_MODES}")
    if not 1 <= max_depth <= len(ADVISOR_KINDS):
        raise ValueError(f"max_depth must be in [1, {len(ADVISOR_KINDS)}]")
    eid, keys, gold = int(row["example_id"]), list(row["choices"]), row["ground_truth"]
    qhash = question_hash(row["question"])
    available = [k for k in ADVISOR_KINDS if pool.has(k)]
    base = protocol.manager_messages(bench, row)
    checks, policy_action = [], None
    if root_mode == "policy":
        root = manager.root(base, keys, available, eid)
        direct_pred, direct_text, direct_valid = root.key, root.text, True
        checks.append(root.check)
        policy_action = {"action": root.name, "draft": root.key, "logprob": round(root.logprob, 6),
                         "key_logprobs": root.key_logprobs}
    else:
        direct_pred, direct_text, direct_valid = manager.probe(base, keys, _DIRECT_PROBE)
    direct_correct = bool(direct_valid and direct_pred == gold)

    branches: List[Dict[str, Any]] = []
    if direct_valid and direct_pred is not None:
        frontier = [_State((), list(base), [direct_pred], [])]
        for depth in range(1, max_depth + 1):
            specs = [(s, k) for s in frontier for k in available if k not in s.sequence]
            if hasattr(pool, "prefetch"):  # concurrent advisor requests for the whole layer
                pool.prefetch([AdvisorRequest.for_row(k, row, s.drafts[-1]) for s, k in specs])
            states = []
            for state, kind in specs:
                draft, sequence = state.drafts[-1], state.sequence + (kind,)
                call_id = f"mv_{eid}_{'_'.join(sequence)}"
                call = protocol.call_message(kind, draft, eid, call_id)
                shown = protocol.eval_call_message(kind, draft, eid, call_id) if root_mode == "policy" else call
                tool = protocol.tool_message(kind, call_id, _advise(pool, row, kind, draft))
                event = {"role": "assistant", "content": call["content"], "tool_calls": call["tool_calls"],
                         "tool_event": tool}
                states.append((state, sequence, state.messages + [shown, tool], state.trajectory + [event]))
            if root_mode == "policy":
                decisions = manager.revise([m for _, _, m, _ in states], keys)
                revisions = [(d.key, d.text, True) for d in decisions]
                checks += [d.check for d in decisions]
            else:
                decisions = [None] * len(states)
                revisions = [manager.probe(m, keys, _REVISION_PROBE) for _, _, m, _ in states]
            next_frontier, depth_success = [], False
            for (state, sequence, messages, trajectory), (pred, text, valid), decision in zip(states, revisions, decisions):
                drafts = state.drafts + [pred if pred is not None else state.drafts[-1]]
                correct = bool(valid and pred == gold)
                branch = {
                    "example_id": eid, "question_hash": qhash, "sequence": list(sequence), "depth": len(sequence),
                    "initial_draft": direct_pred, "drafts": drafts, "final_pred": pred, "valid": valid,
                    "correct": correct, "probe_text": text[:TEXT_CAP], "trajectory": trajectory,
                }
                if decision is not None:
                    branch["revision"] = {"draft": decision.draft or pred, "logprob": round(decision.logprob, 6),
                                          "key_logprobs": decision.key_logprobs,
                                          "answer_logprobs": decision.answer_logprobs,
                                          "would_call": bool(decision.check.get("would_call"))}
                branches.append(branch)
                next_frontier.append(_State(sequence, messages, drafts, trajectory))
                depth_success = depth_success or correct
            if direct_correct or depth_success:  # marginal_value.py:624
                break
            frontier = next_frontier

    preferred = choose_preferred_sequence(direct_correct, branches, tie_break_seed=seed + eid)
    return {
        "example_id": eid,
        "question_hash": qhash,
        "benchmark_name": row.get("benchmark_name", bench.name),
        "ground_truth": gold,
        "direct_pred": direct_pred,
        "direct_valid": direct_valid,
        "direct_correct": direct_correct,
        "direct_text": direct_text[:TEXT_CAP],
        "preferred_sequence": list(preferred) if preferred is not None else None,
        "base_messages": base,
        "branches": branches,
        "split": "train",
        "root_mode": root_mode,
        "policy_action": policy_action,
        "unconstrained_argmax": protocol.merge_checks(checks) if root_mode == "policy" else None,
    }


def collect(rows: Sequence[Dict[str, Any]], bench, manager, pool, out_dir, *, pool_name: str, round_index: int,
            root_mode: str = "policy", max_depth: int = 2, seed: int = DEFAULT_SEED, resume: bool = False,
            lookahead: int = 32) -> Dict[str, Any]:
    """Collect every row into ``out_dir/shards/<example_id>.json`` and merge the paper output files.

    With a prefetching pool, the draft-free Extractor/Reasoner requests of the next
    ``lookahead`` roots run in the background while the manager works; a failed
    request surfaces (fail-stop) before its root is processed. Any failure aborts
    the pool (no further attempts) and cancels queued lookahead work without
    waiting for it; HTTP attempts already in flight finish within the pool
    ``timeout`` (their outputs are valid cache entries).
    """
    out = Path(out_dir)
    eids = [int(r["example_id"]) for r in rows]
    if len(set(eids)) != len(eids):
        raise ValueError("duplicate example_id in the root pool")
    identity = {
        "collector_version": COLLECTOR_VERSION, "benchmark": bench.name, "pool": pool_name, "round": int(round_index),
        "root_mode": root_mode, "max_depth": int(max_depth), "seed": int(seed), "rows_sha256": rows_digest(rows),
        "manager": manager.identity(), "advisors": pool.identity() if hasattr(pool, "identity") else None,
        "harness": harness_identity(), "source_sha256": source_sha256(),
    }
    manifest, shards = out / "collect_manifest.json", out / "shards"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{out} already holds a collection; pass resume=True to continue it")
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != json.loads(json.dumps(identity)):
            diff = sorted(k for k in set(old) | set(identity) if old.get(k) != json.loads(json.dumps(identity)).get(k))
            raise ValueError(f"{out}: resume identity differs in {diff}")
        _check_shards(shards, rows, identity)
    else:
        if shards.is_dir() and any(shards.glob("*.json")):
            raise FileExistsError(f"{shards} holds shards without a collect_manifest.json; refusing to adopt them")
        _atomic_write(manifest, json.dumps(identity, indent=2, sort_keys=True) + "\n")
    shard = lambda eid: shards / f"{eid}.json"
    todo = [r for r in rows if not shard(int(r["example_id"])).exists()]
    draft_free = [k for k in ADVISOR_KINDS if k != "verifier" and pool.has(k)]
    ahead = ThreadPoolExecutor(max(1, getattr(pool, "workers", 2) // 2)) \
        if lookahead > 0 and hasattr(pool, "prefetch") else None
    pending: Dict[int, Any] = {}
    if hasattr(pool, "clear_abort"):
        pool.clear_abort()
    try:
        from tqdm import tqdm
        iterator = tqdm(todo, desc=f"mcq_rsi collect {bench.name}/{pool_name}", unit="q")
    except ImportError:
        iterator = todo
    try:
        for i, row in enumerate(iterator):
            if ahead is not None:
                for r in todo[i:i + lookahead]:
                    if int(r["example_id"]) not in pending:
                        pending[int(r["example_id"])] = ahead.submit(
                            pool.prefetch, [AdvisorRequest.for_row(k, r) for k in draft_free])
                pending.pop(int(row["example_id"])).result()
            record = collect_question(row, bench, manager, pool, root_mode=root_mode, max_depth=max_depth, seed=seed)
            record.update(pool=pool_name, round=int(round_index))
            _atomic_write(shard(record["example_id"]), _dumps(record) + "\n")
    except BaseException:
        if hasattr(pool, "abort"):  # stop outstanding lookahead retries now
            pool.abort()
        raise
    finally:
        if ahead is not None:
            ahead.shutdown(wait=False, cancel_futures=True)
    return merge(out, eids, identity)


def merge(out: Path, eids: Sequence[int], identity: Dict[str, Any]) -> Dict[str, Any]:
    records = [json.loads((out / "shards" / f"{eid}.json").read_text(encoding="utf-8")) for eid in eids]
    report = summarize_counterfactuals(records)
    policy = [r["policy_action"] for r in records if r.get("policy_action")]
    report.update({
        "benchmark": identity["benchmark"], "pool": identity["pool"], "round": identity["round"],
        "root_mode": identity["root_mode"], "max_depth": identity["max_depth"], "seed": identity["seed"],
        "n_branches": sum(len(r["branches"]) for r in records),
        "policy_action_counts": {a: sum(p["action"] == a for p in policy) for a in ("commit", *ADVISOR_KINDS)},
        "policy_call_rate": sum(p["action"] != "commit" for p in policy) / max(1, len(policy)),
        "unconstrained_argmax": protocol.merge_checks(r.get("unconstrained_argmax") for r in records),
        "revision_answer_draft_mismatch": sum(b["revision"]["draft"] != b["final_pred"] for r in records
                                              for b in r["branches"] if b.get("revision")),
    })
    _atomic_write(out / "counterfactual_records.jsonl", "".join(_dumps(r) + "\n" for r in records))
    _atomic_write(out / "counterfactual_branches.jsonl",
                  "".join(_dumps(b) + "\n" for r in records for b in r["branches"]))
    _atomic_write(out / "marginal_value_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"records_jsonl": str(out / "counterfactual_records.jsonl"),
            "report_json": str(out / "marginal_value_report.json"), "report": report}


def policy_turns(eval_row: Dict[str, Any], keys) -> List[Optional[Tuple[str, Optional[str]]]]:
    """Grammar actions ``(draft, kind|None)`` of a paper-era ``manager_tool_eval.jsonl`` trajectory, in order."""
    return [protocol.parse_decision(e.get("content"), e.get("tool_call"), keys)
            for e in eval_row.get("trajectory", []) if e.get("role") == "assistant"]


def policy_action_from_eval(eval_row: Dict[str, Any], keys) -> Optional[Dict[str, Any]]:
    """The root ``policy_action`` a deployed paper-era eval trajectory implies (no scores)."""
    turns = policy_turns(eval_row, keys)
    if not turns or turns[0] is None:
        return None
    key, kind = turns[0]
    return {"action": kind or "commit", "draft": key, "logprob": None, "key_logprobs": None}

