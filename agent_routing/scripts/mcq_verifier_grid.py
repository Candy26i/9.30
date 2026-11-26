#!/usr/bin/env python3
"""Equal-information delegation grid (reviewer-requested control, 2026-11).

The submission's always-Verifier baseline gave the Verifier no candidate answer, while a learned policy call does.
This script measures every action from the SAME manager-generated candidate, with the same manager checkpoint,
decoding, revision prompt and advisor cache, so that only the calling rule differs.

For every question of one pool and one manager checkpoint:

  root              the manager's own grammar-constrained greedy deployment turn (the collector's
                    ``root_mode="policy"``): candidate X, the policy's action, its log-probability and the
                    draft-key distribution (confidence)
  extractor         forced Extractor call from X, then the forced-commit revision      } exactly the collector's
  reasoner          forced Reasoner call from X, ...                                   } depth-1 branches
  verifier          forced Verifier call WITH the candidate (``current_draft = X``)    } (``collect_question``)
  verifier_nocand   the Verifier call with the candidate removed: the ``current_draft`` argument is "" and the
                    Verifier's prompt has no candidate block (it answers "no candidate provided")
  null              the call turn of ``verifier`` unchanged, tool output replaced by an empty Verifier response
                    (every field empty): a revision pass with no specialist evidence (manager-only revision)

The first four are ``collect.collect_question(..., max_depth=1)``, so they are byte-compatible with the RSI collection
records; the two extra branches reuse the same message builders and ``Manager.revise`` (commit forced, outcome Z of
``DRAFT_ANSWER_Y\\nANSWER_Z``) and are stored under ``extra_branches``.

One atomic shard per question; ``--resume`` continues a run with the same identity. Output:
``<out>/grid_records.jsonl``, ``<out>/grid_summary.json`` (see ``mcq_verifier_grid_analyze.py`` for the statistics).

    python scripts/mcq_verifier_grid.py --config configs/mcq_rsi_medqa_v3.json --pool test \\
        --checkpoint /workspace/mcq_rsi/import/medqa/round1/sft --out /workspace/mcq_rsi/runs/verifier_grid/s1/medqa/test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GRID_VERSION = "mcq_verifier_grid/2"
EXTRA_KINDS = ("verifier_nocand", "null")
# An empty Verifier response: the Verifier's JSON schema with every field empty and no confidence value.
NULL_OUTPUT = json.dumps({"relevant_principles": [], "checks": [], "potential_errors": [],
                          "candidate_answer_audit": "", "uncertainty_notes": []})


def extra_states(row: Dict[str, Any], record: Dict[str, Any], pool, kinds: Sequence[str]):
    """(kind, call turn shown to the manager, tool message, revision state) for each extra branch of one root."""
    from src.manager.mcq_rsi import protocol

    eid, draft, base = int(row["example_id"]), record["direct_pred"], record["base_messages"]
    out = []
    for kind in kinds:
        if kind not in EXTRA_KINDS:
            raise ValueError(f"unknown extra branch {kind!r} (known: {EXTRA_KINDS})")
        call_id = f"vg_{eid}_{kind}"
        shown = protocol.eval_call_message("verifier", draft, eid, call_id)  # as the deployed eval keeps a Verifier call
        if kind == "verifier_nocand":
            function = shown["tool_calls"][0]["function"]
            arguments = json.loads(function["arguments"])
            if "current_draft" not in arguments:
                raise ValueError("the Verifier call turn carries no current_draft argument")
            arguments["current_draft"] = ""
            function["arguments"] = json.dumps(arguments, ensure_ascii=False)
            output = pool.call(agent_kind="verifier", example_id=eid, question=row["question"],
                               context=row.get("context") or "", choices=row["choices"],
                               cache_namespace="verifier_grid", candidate_answer="")
        else:
            output = NULL_OUTPUT
        tool = protocol.tool_message("verifier", call_id, output)
        out.append((kind, shown, tool, list(base) + [shown, tool]))
    return out


class _RootTap:
    """The manager, keeping the check of its last root turn: the grammar-constrained path against unconstrained greedy.

    ``mismatch == 0`` means the constrained first turn is what plain greedy decoding (the deployed evaluation) writes.
    That holds for the trained managers; the untrained base model does not follow the DRAFT/commit/call grammar on its
    own, so its grid describes a grammar-forced system and the analysis says so.
    """

    def __init__(self, manager):
        self._manager, self.check = manager, None

    def __getattr__(self, name):
        return getattr(self._manager, name)

    def root(self, *args, **kwargs):
        decision = self._manager.root(*args, **kwargs)
        self.check = dict(decision.check or {})
        return decision


def grid_question(row: Dict[str, Any], bench, manager, pool, extra: Sequence[str] = EXTRA_KINDS,
                  seed: int = 42) -> Dict[str, Any]:
    """The collector's depth-1 record of one root plus the extra branches from the same candidate."""
    from src.manager.mcq_rsi import collect

    tap = _RootTap(manager)
    record = collect.collect_question(row, bench, tap, pool, root_mode="policy", max_depth=1, seed=seed)
    record["root_check"] = tap.check
    record["extra_branches"] = []
    if not extra or not record["direct_valid"] or record["direct_pred"] is None:
        return record
    if not pool.has("verifier"):
        raise ValueError("the extra branches need the Verifier")
    keys, gold, draft = list(row["choices"]), row["ground_truth"], record["direct_pred"]
    states = extra_states(row, record, pool, extra)
    decisions = manager.revise([s for _, _, _, s in states], keys)
    for (kind, shown, tool, _), d in zip(states, decisions):
        record["extra_branches"].append({
            "example_id": int(row["example_id"]), "kind": kind, "sequence": [kind], "depth": 1,
            "initial_draft": draft, "drafts": [draft, d.key], "final_pred": d.key, "valid": True,
            "correct": bool(d.key == gold), "probe_text": d.text[:collect.TEXT_CAP],
            "call": shown, "tool_output_sha256": hashlib.sha256(tool["content"].encode("utf-8")).hexdigest(),
            "tool_output_chars": len(tool["content"]),
            "revision": {"draft": d.draft or d.key, "logprob": round(d.logprob, 6), "key_logprobs": d.key_logprobs,
                         "answer_logprobs": d.answer_logprobs, "would_call": bool(d.check.get("would_call"))},
        })
    return record


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline accuracies of a finished grid (the analysis script has the statistics)."""
    n = len(records)
    if not n:
        return {"n": 0}
    by_kind: Dict[str, List[bool]] = {}
    policy, calls = [], 0
    for r in records:
        one = {b["sequence"][0]: bool(b["correct"]) for b in r["branches"] if len(b["sequence"]) == 1}
        for k, v in one.items():
            by_kind.setdefault(k, []).append(v)
        for b in r.get("extra_branches") or []:
            by_kind.setdefault(b["kind"], []).append(bool(b["correct"]))
        action = (r.get("policy_action") or {}).get("action", "commit")
        calls += action != "commit"
        policy.append(bool(r["direct_correct"]) if action == "commit" else one.get(action, False))
    acc = lambda xs: round(sum(xs) / len(xs), 6) if xs else None
    checks = [r.get("root_check") for r in records]
    return {"n": n, "candidate_accuracy": acc([bool(r["direct_correct"]) for r in records]),
            "always": {k: {"accuracy": acc(v), "n": len(v)} for k, v in sorted(by_kind.items())},
            "policy_single_call": {"accuracy": acc(policy), "call_rate": round(calls / n, 6)},
            # questions whose constrained first turn is exactly what unconstrained greedy decoding writes
            "root_equals_unconstrained_greedy": {"n_equal": sum(bool(c) and c.get("mismatch") == 0 for c in checks),
                                                 "n_checked": sum(bool(c) for c in checks), "n": n}}


def run_grid(rows: Sequence[Dict[str, Any]], bench, manager, pool, out_dir, *, pool_name: str,
             extra: Sequence[str] = EXTRA_KINDS, seed: int = 42, resume: bool = False, lookahead: int = 32,
             progress: bool = True) -> Dict[str, Any]:
    from src.manager.mcq_rsi import collect
    from src.manager.mcq_rsi.advisors import AdvisorRequest
    from src.verifiable.provenance import harness_identity

    out = Path(out_dir)
    for kind in extra:
        if kind not in EXTRA_KINDS:
            raise ValueError(f"unknown extra branch {kind!r} (known: {EXTRA_KINDS})")
    eids = [int(r["example_id"]) for r in rows]
    if len(set(eids)) != len(eids):
        raise ValueError("duplicate example_id in the pool")
    identity = {
        "grid_version": GRID_VERSION, "collector_version": collect.COLLECTOR_VERSION, "benchmark": bench.name,
        "pool": pool_name, "seed": int(seed), "extra": list(extra), "rows_sha256": collect.rows_digest(rows),
        "null_output_sha256": hashlib.sha256(NULL_OUTPUT.encode("utf-8")).hexdigest(),
        "manager": manager.identity(), "advisors": pool.identity() if hasattr(pool, "identity") else None,
        "harness": harness_identity(), "source_sha256": collect.source_sha256(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    identity = json.loads(json.dumps(identity))
    manifest, shards = out / "grid_manifest.json", out / "shards"
    if manifest.exists():
        if not resume:
            raise FileExistsError(f"{out} already holds a grid; pass --resume to continue it")
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != identity:
            diff = sorted(k for k in set(old) | set(identity) if old.get(k) != identity.get(k))
            raise ValueError(f"{out}: resume identity differs in {diff}")
    else:
        if shards.is_dir() and any(shards.glob("*.json")):
            raise FileExistsError(f"{shards} holds shards without a grid_manifest.json; refusing to adopt them")
        collect._atomic_write(manifest, json.dumps(identity, indent=2, sort_keys=True) + "\n")
    shard = lambda eid: shards / f"{eid}.json"
    known = set(eids)
    for p in shards.glob("*.json") if shards.is_dir() else []:
        if not p.stem.isdigit() or int(p.stem) not in known:
            raise ValueError(f"{p}: a shard that is not in this pool")
    todo = [r for r in rows if not shard(int(r["example_id"])).exists()]
    # Draft-free advisor requests of the next roots run in the background: E, R and the candidate-free Verifier.
    draft_free = [k for k in ("extractor", "reasoner") if pool.has(k)]
    if "verifier_nocand" in extra and pool.has("verifier"):
        draft_free.append("verifier")
    ahead = ThreadPoolExecutor(max(1, getattr(pool, "workers", 2) // 2)) \
        if lookahead > 0 and hasattr(pool, "prefetch") else None
    pending: Dict[int, Any] = {}
    if hasattr(pool, "clear_abort"):
        pool.clear_abort()
    iterator = todo
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(todo, desc=f"verifier grid {bench.name}/{pool_name}", unit="q")
        except ImportError:
            pass
    try:
        for i, row in enumerate(iterator):
            if ahead is not None:
                for r in todo[i:i + lookahead]:
                    if int(r["example_id"]) not in pending:
                        pending[int(r["example_id"])] = ahead.submit(
                            pool.prefetch, [AdvisorRequest.for_row(k, r) for k in draft_free])
                pending.pop(int(row["example_id"])).result()
            record = grid_question(row, bench, manager, pool, extra=extra, seed=seed)
            record.update(pool=pool_name, split=pool_name)
            collect._atomic_write(shard(record["example_id"]), json.dumps(record, ensure_ascii=False) + "\n")
    except BaseException:
        if hasattr(pool, "abort"):
            pool.abort()
        raise
    finally:
        if ahead is not None:
            ahead.shutdown(wait=False, cancel_futures=True)
    records = [json.loads(shard(eid).read_text(encoding="utf-8")) for eid in eids]
    for r, eid in zip(records, eids):
        if int(r["example_id"]) != eid or r.get("pool") != pool_name:
            raise ValueError(f"shard {eid} does not belong to this run")
    collect._atomic_write(out / "grid_records.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    summary = {**summarize(records), "benchmark": bench.name, "pool": pool_name, "extra": list(extra),
               "manager": identity["manager"], "grid_version": GRID_VERSION}
    collect._atomic_write(out / "grid_summary.json", json.dumps(summary, indent=2) + "\n")
    return {"records_jsonl": str(out / "grid_records.jsonl"), "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/mcq_rsi_<bench>_v3.json")
    ap.add_argument("--pool", required=True, help="split-manifest pool: dev, test, test_paper, ...")
    ap.add_argument("--checkpoint", required=True, help="manager adapter directory, or the base snapshot directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default=",".join(EXTRA_KINDS), help="comma list of extra branches ('' = none)")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke runs)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--offline", action="store_true", help="advisor cache only (no server, no preflight check)")
    args = ap.parse_args()

    from src.manager.mcq_rsi import controller

    cfg = controller.load_config(args.config)
    if args.offline:
        cfg["advisor_url"] = None
    rt = controller.Runtime(cfg)
    rows = rt.rows(args.pool)
    if args.limit:
        rows = rows[:args.limit]
    extra = [k for k in args.extra.split(",") if k]
    started = time.time()
    pool = rt.pool()  # checks the server and the preflight record when a server URL is configured
    result = run_grid(rows, rt.bench, rt.manager(args.checkpoint), pool, args.out, pool_name=args.pool, extra=extra,
                      seed=int(cfg["collect"]["seed"]), resume=args.resume)
    s = result["summary"]
    always = " ".join(f"{k}={v['accuracy']:.3f}" for k, v in s["always"].items())
    print(f"[VERIFIER_GRID] {rt.bench.name}/{args.pool} n={s['n']} candidate={s['candidate_accuracy']:.3f} "
          f"policy={s['policy_single_call']['accuracy']:.3f}@{s['policy_single_call']['call_rate']:.3f} always: {always} "
          f"root=greedy on {s['root_equals_unconstrained_greedy']['n_equal']}/{s['n']} "
          f"({time.time() - started:.0f}s) advisor_stats={getattr(pool, 'stats', None)} -> {result['records_jsonl']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
