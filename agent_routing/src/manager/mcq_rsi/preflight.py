"""Advisor-server preflight (design §7.2 steps 1-2): vLLM identity + LoRA-actually-applied replay gate.

1. Identity: ``GET /version`` is recorded and ``CachedAdvisorPool.check_server`` must pass
   (every ``<bench>_<kind>`` LoRA served on ``Qwen/Qwen3.5-9B`` from the pinned adapter, or
   from a verified renamed copy, ``serving``).
2. Replay: N recorded paper advisor outputs per kind (the round-1 tree's ``tool_event``
   contents, Verifier with its recorded ``current_draft``) are regenerated greedily and
   sequentially with the request the RSI cache will send. The paper servers ran on another
   GPU (~96 GB, SM 12.x) with other vLLM flags, and bf16 greedy decoding drifts across
   GPUs and kernels somewhere in a long output, so exact match is not required: a kind
   passes if ``exact-match rate >= min_match`` (0.9) **or** the LoRA output is closer
   (word-level similarity) to the recorded output than the base output on
   ``>= min_closer`` of the items with median LoRA similarity ``>= min_similarity``.
   Exact match, first-divergence (common-prefix) ratios and similarities are all reported.
3. LoRA effect: the same requests are sent to the served base model; the LoRA output
   must differ from the base output on ``>= min_lora_effect`` of them. A vLLM server that
   silently ignores every LoRA tensor (PEFT ``model.layers.*`` keys on the multimodal
   architecture) answers identically for both and fails here even when the replay passes.

A failing kind gets a diagnosis (LoRA not applied / paper outputs closer to the plain
base (D14) / prompt or decoding mismatch); the reasoner is also replayed under the 9.30
runtime prompt when it differs from the registry prompt (the OPEN question which
Reasoner prompt was live for GPQA, MMLU-Pro and AQuA), as information. The report records
the vLLM ``/version``, the GPU and the server flags fingerprint that
``scripts/start_mcq_advisors.sh`` writes; ``require_passed`` refuses a server whose
version or flags differ from the passing report. Nothing is written to the advisor cache.
``write_report`` stores the result where ``Runtime.pool`` requires it before the
controller fetches anything.
"""
from __future__ import annotations

import difflib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..marginal_value import ADVISOR_KINDS
from . import benchmarks as registry

PREFLIGHT_VERSION = "mcq_rsi_preflight/1"


def recorded_items(records: Sequence[Dict[str, Any]], rows_by_id: Dict[int, Dict[str, Any]], kind: str, n: int,
                   seed: int = 0) -> List[Dict[str, Any]]:
    """Up to ``n`` distinct recorded (row, candidate, output) items of ``kind`` from round-1 records."""
    seen, items = set(), []
    for rec in records:
        row = rows_by_id.get(int(rec["example_id"]))
        if row is None:
            continue
        for branch in rec.get("branches", []):
            for event in branch.get("trajectory", []):
                tool = event.get("tool_event") or {}
                if tool.get("name") != f"{kind}_tool":
                    continue
                args = json.loads(event["tool_calls"][0]["function"].get("arguments") or "{}")
                candidate = str(args.get("current_draft") or "") if kind == "verifier" else ""
                key = (int(rec["example_id"]), candidate)
                if key in seen or not str(tool.get("content") or "").strip():
                    continue
                seen.add(key)
                items.append({"example_id": key[0], "candidate": candidate, "output": tool["content"], "row": row})
    items.sort(key=lambda x: (x["example_id"], x["candidate"]))
    return random.Random(f"preflight:{kind}:{seed}").sample(items, min(n, len(items)))


def _post(http, url: str, payload: Dict[str, Any], timeout: int) -> str:
    resp = http.post(f"{url}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    return str(resp.json()["choices"][0]["message"]["content"] or "").strip()


def similarity(a: str, ref: str) -> float:
    """Word-level edit similarity (difflib ratio) of two outputs; 1.0 iff identical word sequences."""
    wa, wb = str(a).split(), str(ref).split()
    if not wa and not wb:
        return 1.0
    return difflib.SequenceMatcher(None, wa, wb, autojunk=False).ratio()


def prefix_ratio(a: str, ref: str) -> float:
    """Share of ``ref`` (characters) before the first divergence of ``a`` from it."""
    return len(os.path.commonprefix([str(a), str(ref)])) / max(1, len(str(ref)))


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def diagnose(kind_result: Dict[str, Any], min_match: float, min_effect: float, min_closer: float = 0.75,
             min_similarity: float = 0.5) -> Optional[str]:
    if kind_result["n"] == 0:
        return "no recorded outputs to replay"
    effect, match, base = kind_result["lora_effect_rate"], kind_result["match_rate"], kind_result["base_match_rate"]
    if effect < min_effect:
        return ("LoRA not applied: the LoRA and base answers are identical (plain-base serving trap); serve the "
                "renamed copies (start_mcq_advisors.sh LORA_MODE=multimodal) and rerun")
    if match >= min_match:
        return None
    if (kind_result.get("lora_closer_rate", 0.0) >= min_closer
            and kind_result.get("median_lora_similarity", 0.0) >= min_similarity):
        return None  # numeric drift: the LoRA reproduces the recorded outputs better than the base does
    if base >= min_match or (kind_result.get("base_closer_rate", 0.0) >= min_closer
                             and kind_result.get("median_base_similarity", 0.0) >= min_similarity):
        return ("the recorded paper outputs are closer to the plain base than to the LoRA: the paper-era server did "
                "not apply the advisor LoRAs; decision D14 in docs/MCQ_RSI_RUNBOOK.md (do not run until decided)")
    return ("LoRA applied but the recorded outputs are not reproduced (neither exactly nor closer than the base): "
            "check the prompt registry (reasoner variant rates), vLLM version, dtype and thinking settings")


def diagnose_base(kind_result: Dict[str, Any], min_match: float, min_similarity: float = 0.5) -> Optional[str]:
    """Base advisors (advisor_mode "base") must reproduce the recorded paper-era outputs, which came from the
    base model (D14): exactly, or as closely as bf16 greedy drift allows (median similarity >= min_similarity;
    the trained adapters score 0.17-0.30 against them, the base 0.63-0.83 on MedQA)."""
    if kind_result["n"] == 0:
        return "no recorded outputs to replay"
    if kind_result["match_rate"] >= min_match or kind_result["median_base_similarity"] >= min_similarity:
        return None
    return ("the base model does not reproduce the recorded paper advisor outputs: check the prompt registry, "
            "chat template (thinking off), vLLM version, dtype and the pinned base revision")


def _replay_base(http, url, pool, kind, items, min_match, min_similarity, progress) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"n": len(items), "match": 0, "examples": []}
    sims, prefixes = [], []
    for i, item in enumerate(items):
        r = item["row"]
        recorded = item["output"].strip()
        base = _post(http, url, pool.request_payload(kind, r["question"], r.get("context") or "", r["choices"],
                                                     item["candidate"]), pool.timeout)
        sims.append(similarity(base, recorded))
        prefixes.append(prefix_ratio(base, recorded))
        stats["match"] += base == recorded
        if len(stats["examples"]) < 3:
            stats["examples"].append({"example_id": item["example_id"], "candidate": item["candidate"],
                                      "base_equal": base == recorded, "base_similarity": sims[-1],
                                      "base_prefix_ratio": prefixes[-1], "recorded_head": recorded[:160],
                                      "base_head": base[:160]})
        if progress:
            progress(f"{kind} {i + 1}/{len(items)} match={stats['match']} (base advisors)")
    n = max(1, stats["n"])
    stats.update(match_rate=stats["match"] / n, base_match_rate=stats["match"] / n,
                 median_base_similarity=_median(sims), median_base_prefix_ratio=_median(prefixes))
    stats["diagnosis"] = diagnose_base(stats, min_match, min_similarity)
    stats["replay"] = ("exact" if stats["match_rate"] >= min_match else "similar") if stats["diagnosis"] is None else "failed"
    stats["passed"] = stats["diagnosis"] is None
    return stats


def server_version(pool, http=None) -> Dict[str, Any]:
    try:
        resp = (http or pool._client()).get(f"{pool.server_url}/version", timeout=pool.timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:  # noqa: BLE001 - recorded; identity is checked by check_server
        return {"error": str(e)}


def server_flags(pool) -> Optional[Dict[str, Any]]:
    """The flags fingerprint ``start_mcq_advisors.sh`` wrote next to the served LoRAs
    (``server_flags_<port>.json``; ``MCQ_ADVISOR_FLAGS_FILE`` overrides), or None."""
    env = os.environ.get("MCQ_ADVISOR_FLAGS_FILE")
    port = urlparse(pool.server_url or "").port
    candidates = [Path(env)] if env else [Path(v["root"]).parent / f"server_flags_{port}.json"
                                           for v in (pool.served or {}).values()
                                           if v.get("root") and "model" not in v]
    if not env and getattr(pool, "mode", "lora") == "base":  # no adapter directory: the script's default one
        candidates.append(Path(os.environ.get("MCQ_SERVED_LORAS", "/workspace/mcq_rsi/served_loras"))
                          / f"server_flags_{port}.json")
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except ValueError:
                return {"unreadable": str(path)}
    return None


def gpu_info() -> Optional[List[str]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return [line.strip() for line in out.stdout.splitlines() if line.strip()] or None
    except (OSError, subprocess.SubprocessError):
        return None


def run_preflight(bench_name: str, pool, records: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], *,
                  n_per_kind: int = 20, min_match: float = 0.9, min_lora_effect: float = 0.5, min_closer: float = 0.75,
                  min_similarity: float = 0.5, seed: int = 0, http=None, kinds: Sequence[str] = ADVISOR_KINDS,
                  progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    from ...subagents.prompts.runtime_prompts import build_runtime_messages
    from . import prompts
    bench = registry.get(bench_name)
    url = pool.server_url
    if not url:
        raise ValueError("preflight needs the advisor server URL")
    http = http or pool._client()
    started = time.time()
    version = server_version(pool, http)
    served = pool.check_server()
    rows_by_id = {int(r["example_id"]): r for r in rows}
    settings = {"n_per_kind": n_per_kind, "min_match": min_match, "min_lora_effect": min_lora_effect,
                "min_closer": min_closer, "min_similarity": min_similarity, "seed": seed}
    result: Dict[str, Any] = {"version": PREFLIGHT_VERSION, "bench": bench.name, "server_url": url,
                              "vllm_version": version, "served_models": served, "verified_loras": pool.served,
                              "server_flags": server_flags(pool), "gpu": gpu_info(),
                              "pool_identity": pool.identity(), "settings": settings, **settings, "kinds": {}}
    base_mode = getattr(pool, "mode", "lora") == "base"
    if base_mode:
        result["advisor_mode"] = "base"
    for kind in kinds:
        items = recorded_items(records, rows_by_id, kind, n_per_kind, seed)
        if base_mode:
            result["kinds"][kind] = _replay_base(http, url, pool, kind, items, min_match, min_similarity, progress)
            continue
        variant = None
        runtime_system = build_runtime_messages(kind, "q", "", {"A": "a", "B": "b"})[0]["content"]
        if kind == "reasoner" and runtime_system != prompts.system_prompt(bench.name, kind):
            variant = runtime_system
        stats = {"n": len(items), "match": 0, "base_match": 0, "lora_effect": 0, "variant_match": 0,
                 "lora_closer": 0, "base_closer": 0, "examples": []}
        sims: Dict[str, List[float]] = {"lora": [], "base": [], "lora_prefix": [], "base_prefix": [], "variant": []}
        for i, item in enumerate(items):
            r = item["row"]
            args = (kind, r["question"], r.get("context") or "", r["choices"], item["candidate"])
            recorded = item["output"].strip()
            lora = _post(http, url, pool.request_payload(*args), pool.timeout)
            base = _post(http, url, pool.request_payload(*args, model=registry.BASE_MODEL), pool.timeout)
            ls, bs = similarity(lora, recorded), similarity(base, recorded)
            sims["lora"].append(ls)
            sims["base"].append(bs)
            sims["lora_prefix"].append(prefix_ratio(lora, recorded))
            sims["base_prefix"].append(prefix_ratio(base, recorded))
            stats["match"] += lora == recorded
            stats["base_match"] += base == recorded
            stats["lora_effect"] += lora != base
            stats["lora_closer"] += ls > bs
            stats["base_closer"] += bs > ls
            if variant is not None:
                v = _post(http, url, pool.request_payload(*args, system=variant), pool.timeout)
                stats["variant_match"] += v == recorded
                sims["variant"].append(similarity(v, recorded))
            if len(stats["examples"]) < 3:
                stats["examples"].append({"example_id": item["example_id"], "candidate": item["candidate"],
                                          "lora_equal": lora == recorded, "base_equal": base == recorded,
                                          "lora_similarity": ls, "base_similarity": bs,
                                          "lora_prefix_ratio": sims["lora_prefix"][-1],
                                          "recorded_head": recorded[:160], "lora_head": lora[:160]})
            if progress:
                progress(f"{kind} {i + 1}/{len(items)} match={stats['match']} lora_closer={stats['lora_closer']} "
                         f"lora_effect={stats['lora_effect']}")
        n = max(1, stats["n"])
        stats.update(match_rate=stats["match"] / n, base_match_rate=stats["base_match"] / n,
                     lora_effect_rate=stats["lora_effect"] / n, lora_closer_rate=stats["lora_closer"] / n,
                     base_closer_rate=stats["base_closer"] / n,
                     median_lora_similarity=_median(sims["lora"]), median_base_similarity=_median(sims["base"]),
                     median_lora_prefix_ratio=_median(sims["lora_prefix"]),
                     median_base_prefix_ratio=_median(sims["base_prefix"]),
                     variant_match_rate=(stats["variant_match"] / n) if variant is not None else None,
                     median_variant_similarity=_median(sims["variant"]) if variant is not None else None)
        stats["diagnosis"] = diagnose(stats, min_match, min_lora_effect, min_closer, min_similarity)
        stats["replay"] = ("exact" if stats["match_rate"] >= min_match else "closer_than_base") \
            if stats["diagnosis"] is None else "failed"
        stats["passed"] = stats["diagnosis"] is None
        result["kinds"][kind] = stats
    result["passed"] = bool(result["kinds"]) and all(k["passed"] for k in result["kinds"].values())
    if base_mode:
        # Base advisors are bound to the pinned base revision only through the server's recorded flags.
        args = (result["server_flags"] or {}).get("args") or []
        if registry.BASE_REVISION not in args:
            result["passed"] = False
            result["flags_problem"] = (f"base advisors need the server flags file with --revision "
                                       f"{registry.BASE_REVISION} (start_mcq_advisors.sh writes it); found {args or None}")
    result["seconds"] = time.time() - started
    result["finished_unix"] = time.time()
    return result


def report_path(advisor_cache, bench: str) -> Path:
    return Path(advisor_cache) / "preflight" / f"{bench}.json"


def write_report(advisor_cache, result: Dict[str, Any]) -> Path:
    from .collect import _atomic_write
    path = report_path(advisor_cache, result["bench"])
    _atomic_write(path, json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def require_passed(advisor_cache, pool) -> Dict[str, Any]:
    """Refuse an advisor server that has not passed preflight with this exact identity and served adapters."""
    path = report_path(advisor_cache, pool.bench)
    if not path.is_file():
        raise RuntimeError(f"no advisor preflight for {pool.bench} ({path}); run `python -m src.manager.mcq_rsi "
                           "preflight` first")
    rep = json.loads(path.read_text(encoding="utf-8"))
    if not rep.get("passed"):
        raise RuntimeError(f"{path}: the advisor preflight failed; see its diagnoses")
    if rep.get("pool_identity") != json.loads(json.dumps(pool.identity())):
        raise RuntimeError(f"{path}: preflight ran for another advisor identity")
    if rep.get("server_url") != pool.server_url or rep.get("verified_loras") != pool.served:
        raise RuntimeError(f"{path}: preflight ran against another server or other served adapters; rerun it")
    live_version = json.loads(json.dumps(server_version(pool)))
    if rep.get("vllm_version") != live_version:
        raise RuntimeError(f"{path}: preflight ran against vLLM {rep.get('vllm_version')}, the server reports "
                           f"{live_version}; rerun preflight")
    live_flags = json.loads(json.dumps(server_flags(pool)))
    if rep.get("server_flags", "missing") != live_flags:
        raise RuntimeError(f"{path}: the advisor server was restarted with other flags ({live_flags}) than the "
                           f"passing preflight ({rep.get('server_flags')}); rerun preflight")
    return rep


def sequential_recheck(pool, requests, n: int, seed: int = 0, http=None) -> Dict[str, Any]:
    """Regenerate up to ``n`` cached outputs per kind one request at a time and compare them with the cache.

    The cache is filled ``pool.workers`` requests at a time; vLLM is not batch-invariant, so batched
    outputs can differ from the sequential ones preflight approved. Reported (S1_dev parity.json),
    not gated. Requests whose output is not cached are skipped.
    """
    if not getattr(pool, "server_url", None):
        return {"skipped": "offline pool"}
    http = http or pool._client()
    by_kind: Dict[str, List[Any]] = {}
    for r in requests:
        by_kind.setdefault(r.kind, []).append(r)
    out: Dict[str, Any] = {"workers": getattr(pool, "workers", None), "n_per_kind": n, "kinds": {}}
    for kind, reqs in sorted(by_kind.items()):
        reqs = sorted({(r.example_id, r.candidate): r for r in reqs}.values(), key=lambda r: (r.example_id, r.candidate))
        sample = random.Random(f"recheck:{kind}:{seed}").sample(reqs, min(n, len(reqs)))
        exact, sims, prefixes = 0, [], []
        for r in sample:
            cached = pool.cached(r)
            if cached is None:
                continue
            fresh = _post(http, pool.server_url, pool.request_payload(r.kind, r.question, r.context, r.choices,
                                                                      r.candidate), pool.timeout)
            exact += fresh == cached.strip()
            sims.append(similarity(fresh, cached))
            prefixes.append(prefix_ratio(fresh, cached))
        m = len(sims)
        out["kinds"][kind] = {"n": m, "exact": exact, "exact_rate": exact / m if m else None,
                              "median_similarity": _median(sims) if m else None,
                              "median_prefix_ratio": _median(prefixes) if m else None}
    return out
