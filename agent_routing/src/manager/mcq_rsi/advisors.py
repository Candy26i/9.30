"""Persistent, concurrent, fail-stop advisor client for MCQ RSI.

``CachedAdvisorPool.call`` has ``RemoteSubagentPool.call``'s contract and request
(greedy, ``max_tokens`` 1024, ``chat_template_kwargs`` at top level, 120 s
timeout) and these differences:

- the vLLM model is the benchmark LoRA ``<bench>_<kind>`` and the system prompt
  is the benchmark's pinned advisor prompt (``prompts.build_advisor_messages``);
- outputs persist on disk, one atomic create-if-absent file per key (written,
  fsynced, then hard-linked into place and the directory fsynced; the first
  writer wins, also across processes; an unreadable or mismatching entry is
  moved aside with a warning and refetched; an empty output is never cached)
  ``sha256(bench, adapter identity, prompt sha, kind, question_hash, input sha,
  candidate if Verifier, decoding)``. ``input sha`` covers question, context and
  choices because same-stem rows with different options exist (see ``splits``);
  ``cache_namespace`` and ``example_id`` are not part of the key;
- ``prefetch`` fetches uncached requests with a thread pool (one request per key
  in flight, also across ``call``); ``check_server`` requires ``/v1/models`` to
  serve every ``<bench>_<kind>`` LoRA on ``registry.BASE_MODEL`` from a directory
  whose ``adapter_model.safetensors`` has the pinned sha256, before the first
  request (the cache key claims that adapter);
- after ``retries`` retries with exponential backoff (an empty output counts as
  a failed attempt) a request raises ``AdvisorError``. The first failure in a
  ``prefetch`` batch cancels its queued requests, stops its running ones'
  retries after their current attempt and raises without waiting for them;
  ``abort()`` does the same for every request of the pool (the collector calls
  it when a run fails). A failure never affects another batch's requests.

Nothing ever returns an ``{"error": ...}`` payload: the paper-era eval swallowed
errors that way and scored 0.735/0.755 instead of 0.825. ``stages.run_eval_manager_tools``
catches ``Exception`` around ``pool.call`` and turns it into exactly such a
payload, so an RSI eval must pass ``FailStopEvalPool(pool)`` (its ``call`` raises
``AdvisorFailStop``, a ``BaseException`` that ``except Exception`` cannot
swallow) and accept the report only if ``eval_gate`` returns ``[]``
(``malformed_tool_calls == 0``, ``valid_answer_rate == 1.0``, no advisor failures).
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ...benchmarks.base import question_hash
from ..marginal_value import ADVISOR_KINDS
from . import benchmarks as registry
from . import prompts

CACHE_VERSION = "mcq_rsi_advisor_cache/1"
MAX_NEW_TOKENS = 1024  # RemoteSubagentPool default, used by every paper-era advisor call
TIMEOUT = 120


class AdvisorError(RuntimeError):
    """An advisor output could not be obtained; callers must stop, never substitute text."""


class AdvisorFailStop(BaseException):
    """``AdvisorError`` re-raised past ``except Exception`` handlers (``FailStopEvalPool``)."""


@dataclass(frozen=True)
class AdvisorRequest:
    kind: str
    example_id: int
    question: str
    context: str
    choices: Dict[str, str] = field(hash=False)
    candidate: str = ""

    @classmethod
    def for_row(cls, kind: str, row: Dict[str, Any], candidate: str = "") -> "AdvisorRequest":
        return cls(kind, int(row["example_id"]), row["question"], row.get("context") or "", dict(row["choices"]),
                   candidate if kind == "verifier" else "")


def adapter_identity(bench: registry.Benchmark, kind: str) -> str:
    adapter = bench.advisor(kind)
    model_sha = next(d for name, _, d in adapter.files if name == "adapter_model.safetensors")
    return f"{adapter.uri}#sha256={model_sha}"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_gate(report: Dict[str, Any], pool=None) -> List[str]:
    """Design §3.6 / §7.1(5): why an eval report cannot be accepted ([] = accept).

    ``malformed_tool_calls`` counts the advisor errors ``stages.run_eval_manager_tools``
    swallowed into ``{"error": ...}`` tool text; any of them fails the gate.
    """
    failures = []
    if report.get("malformed_tool_calls") != 0:
        failures.append(f"malformed_tool_calls={report.get('malformed_tool_calls')}")
    if report.get("valid_answer_rate") != 1.0:
        failures.append(f"valid_answer_rate={report.get('valid_answer_rate')}")
    if pool is not None and getattr(pool, "stats", {}).get("failed", 0):
        failures.append(f"advisor failures={pool.stats['failed']}")
    return failures


class FailStopEvalPool:
    """The pool to hand ``stages.run_eval_manager_tools``: ``call`` raises ``AdvisorFailStop``.

    The stages eval loop catches ``Exception`` from ``pool.call`` and feeds
    ``{"error": ...}`` to the manager as tool output. ``AdvisorFailStop`` derives
    from ``BaseException`` and aborts the eval instead. Every other attribute is
    the wrapped pool's.
    """

    def __init__(self, pool: "CachedAdvisorPool"):
        self._pool = pool

    def call(self, *args, **kwargs) -> str:
        try:
            return self._pool.call(*args, **kwargs)
        except AdvisorError as e:
            self._pool.abort()
            raise AdvisorFailStop(str(e)) from e

    def __getattr__(self, name):
        return getattr(self._pool, name)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class CachedAdvisorPool:
    def __init__(
        self,
        bench: str,
        cache_dir,
        server_url: Optional[str] = None,
        *,
        adapters: Optional[Dict[str, str]] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        timeout: int = TIMEOUT,
        retries: int = 3,
        backoff: float = 2.0,
        workers: int = 32,
        http=None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.spec = registry.get(bench)
        self.bench = self.spec.name
        self.server_url = server_url.rstrip("/") if server_url else None
        self.adapters = {k: (adapters or {}).get(k) or adapter_identity(self.spec, k) for k in ADVISOR_KINDS}
        self.prompt_shas = {k: prompts.prompt_sha256(self.bench, k) for k in ADVISOR_KINDS}
        self.decode = {"temperature": 0.0, "max_tokens": int(max_new_tokens),
                       "chat_template_kwargs": {"enable_thinking": False}}
        self.timeout, self.retries, self.backoff, self.workers = timeout, retries, backoff, workers
        self.root = Path(cache_dir) / self.bench
        self._http = http
        self._sleep = sleep
        self._memory: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._inflight: Dict[str, threading.Event] = {}
        self._checked = False
        self._abort = threading.Event()
        self._call_log: List[Dict[str, Any]] = []
        self.served: Dict[str, Dict[str, str]] = {}
        self.stats = {"hits": 0, "fetched": 0, "retries": 0, "failed": 0, "not_stopped": 0, "corrupt": 0}

    # ------------------------------------------------------------ identity
    def has(self, agent_kind: str) -> bool:
        return agent_kind in ADVISOR_KINDS

    def identity(self) -> Dict[str, Any]:
        return {"cache_version": CACHE_VERSION, "bench": self.bench, "adapters": self.adapters,
                "prompts": self.prompt_shas, "decode": self.decode,
                "lora_names": {k: self.spec.lora_name(k) for k in ADVISOR_KINDS}}

    def _client(self):
        if self._http is None:
            import requests
            self._http = requests
        return self._http

    def check_server(self) -> List[str]:
        """Fail unless ``/v1/models`` serves every ``<bench>_<kind>`` LoRA as the pinned adapter.

        vLLM LoRA cards carry ``root`` (the ``--lora-modules`` directory) and
        ``parent`` (the served base model name, so the server must serve the base
        as ``registry.BASE_MODEL``). The directory must be readable here (same
        pod, design §6) so its ``adapter_model.safetensors`` can be hashed against
        the sha256 that every cache key of that kind claims; a name match alone
        never binds a served adapter to the cache. The verified roots are stored
        in each fetched entry's ``meta.lora``.
        """
        if not self.server_url:
            raise AdvisorError("no advisor server_url (offline cache-only pool)")
        try:
            resp = self._client().get(f"{self.server_url}/v1/models", timeout=self.timeout)
            resp.raise_for_status()
            cards = {str(m["id"]): m for m in resp.json()["data"]}
        except Exception as e:  # noqa: BLE001 - any failure is an identity failure
            raise AdvisorError(f"advisor server identity check failed: {e}") from e
        served = sorted(cards)
        missing = [self.spec.lora_name(k) for k in ADVISOR_KINDS if self.spec.lora_name(k) not in cards]
        if missing:
            raise AdvisorError(f"advisor server {self.server_url} does not serve {missing} (serves {served})")
        verified = {}
        for kind in ADVISOR_KINDS:
            name, card = self.spec.lora_name(kind), cards[self.spec.lora_name(kind)]
            if card.get("parent") != registry.BASE_MODEL:
                raise AdvisorError(f"{name} runs on {card.get('parent')!r}, expected {registry.BASE_MODEL!r}")
            expected = self.adapters[kind].rpartition("#sha256=")[2]
            weights = Path(str(card.get("root") or "")) / "adapter_model.safetensors"
            got = _file_sha256(weights) if card.get("root") and weights.is_file() else None
            if got != expected:
                raise AdvisorError(f"{name} serves {card.get('root')!r} (adapter sha256 {got}), "
                                   f"expected the pinned {self.adapters[kind]}")
            verified[kind] = {"root": os.path.realpath(str(card["root"])), "parent": card["parent"]}
        self.served = verified
        self._checked = True
        return served

    # --------------------------------------------------------------- keys
    def key_fields(self, kind: str, question: str, context: str, choices: Dict[str, str],
                   candidate: str = "") -> Dict[str, Any]:
        if kind not in ADVISOR_KINDS:
            raise ValueError(f"Unknown advisor kind: {kind}")
        fields = {"version": CACHE_VERSION, "bench": self.bench, "adapter": self.adapters[kind],
                  "prompt_sha256": self.prompt_shas[kind], "kind": kind,
                  "question_hash": question_hash(question),
                  "input_sha256": _sha({"question": question, "context": context or "", "choices": dict(choices)}),
                  "decode": self.decode}
        if kind == "verifier":
            fields["candidate"] = str(candidate or "")
        return fields

    def key(self, *args, **kwargs) -> str:
        return _sha(self.key_fields(*args, **kwargs))

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def _read(self, key: str, fields: Dict[str, Any]) -> Optional[str]:
        with self._lock:
            if key in self._memory:
                return self._memory[key]
        path = self._path(key)
        try:
            with open(path, "rb") as f:
                inode = os.fstat(f.fileno()).st_ino
                raw = f.read()
        except FileNotFoundError:
            return None
        try:
            entry, problem = json.loads(raw.decode("utf-8")), None
        except ValueError as e:
            entry, problem = None, f"unreadable ({e})"
        if problem is None and (not isinstance(entry, dict) or entry.get("key") != key
                                or entry.get("fields") != fields):
            problem = "key/fields mismatch"
        if problem is None and (not isinstance(entry.get("output"), str) or not entry["output"].strip()):
            problem = "empty or non-text output"
        if problem is not None:
            self._quarantine(path, inode, problem)
            return None
        with self._lock:
            self._memory[key] = entry["output"]
        return entry["output"]

    def _quarantine(self, path: Path, inode: int, problem: str) -> None:
        """Move a bad entry aside (kept for inspection) so the key is refetched; never trust it.

        Only the file that was read is moved: if another process already replaced
        it with a good entry, that entry stays.
        """
        aside = path.with_name(f"{path.name}.corrupt-{os.getpid()}-{time.time_ns()}")
        try:
            if os.stat(path).st_ino == inode:
                os.replace(path, aside)
        except FileNotFoundError:  # another reader moved it first
            pass
        with self._lock:
            self.stats["corrupt"] += 1
        warnings.warn(f"advisor cache entry {path} is {problem}; moved to {aside.name} and refetching",
                      RuntimeWarning, stacklevel=3)

    def _write(self, key: str, fields: Dict[str, Any], output: str, meta: Dict[str, Any]) -> str:
        """Atomic create-if-absent; returns the stored output (an earlier writer wins, also across processes).

        The entry is fsynced before it is linked into place and the directory is
        fsynced after, so a visible entry is complete and survives a crash.
        """
        if not isinstance(output, str) or not output.strip():
            raise AdvisorError(f"refusing to cache an empty advisor output for key {key}")
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"key": key, "fields": fields, "output": output, "meta": meta},
                          ensure_ascii=False, sort_keys=True) + "\n"
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{key}.", suffix=".part")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            for _ in range(3):
                try:
                    os.link(tmp, path)
                    _fsync_dir(path.parent)
                    break
                except FileExistsError:
                    existing = self._read(key, fields)  # a bad entry is moved aside and None returned
                    if existing is not None:
                        output = existing
                        break
            else:
                raise AdvisorError(f"could not store advisor cache entry {path}")
        finally:
            os.unlink(tmp)
        with self._lock:
            self._memory[key] = output
        return output

    def put(self, kind: str, question: str, context: str, choices: Dict[str, str], output: str,
            candidate: str = "", meta: Optional[Dict[str, Any]] = None) -> str:
        """Seed the cache with a recorded output (parity replay)."""
        fields = self.key_fields(kind, question, context, choices, candidate)
        key = _sha(fields)
        existing = self._read(key, fields)
        if existing is not None and existing != output:
            raise AdvisorError(f"cache already holds a different {kind} output for key {key}")
        if existing is None and self._write(key, fields, output, dict(meta or {"source": "seeded"})) != output:
            raise AdvisorError(f"cache already holds a different {kind} output for key {key}")
        return key

    # ------------------------------------------------------------- fetch
    def _fetch(self, kind: str, question: str, context: str, choices: Dict[str, str], candidate: str,
               stop: Optional[threading.Event] = None) -> Dict[str, Any]:
        if not self.server_url:
            raise AdvisorError(f"{kind} output not cached and the pool is offline")
        if not self._checked:
            self.check_server()
        model = self.spec.lora_name(kind)
        payload = {"model": model,
                   "messages": prompts.build_advisor_messages(self.bench, kind, question, context, choices,
                                                              candidate_answer=candidate),
                   **self.decode}
        last: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            if self._abort.is_set() or (stop is not None and stop.is_set()):
                raise AdvisorError(f"{model}: stopped after an earlier advisor failure")
            if attempt:
                with self._lock:
                    self.stats["retries"] += 1
                self._sleep(self.backoff * 2 ** (attempt - 1))
            try:
                resp = self._client().post(f"{self.server_url}/v1/chat/completions", json=payload, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
                if body.get("model", model) != model:
                    raise AdvisorError(f"server answered with model {body.get('model')!r}, expected {model!r}")
                choice = body["choices"][0]
                text = choice["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise AdvisorError("advisor response has no text content")
                return {"output": text.strip(), "finish_reason": choice.get("finish_reason"), "model": model}
            except Exception as e:  # noqa: BLE001 - retried, then fail-stop below
                last = e
        raise AdvisorError(f"{model} failed after {self.retries + 1} attempts: {last}") from last

    def abort(self) -> None:
        """Make every fetch raise before its next attempt (fail-stop for concurrent requests)."""
        self._abort.set()

    def clear_abort(self) -> None:
        self._abort.clear()

    def _get(self, *args, stop: Optional[threading.Event] = None) -> str:
        try:
            return self._get_unchecked(*args, stop=stop)
        except AdvisorError:
            with self._lock:
                self.stats["failed"] += 1
            raise

    def _get_unchecked(self, kind: str, question: str, context: str, choices: Dict[str, str], candidate: str,
                       stop: Optional[threading.Event] = None) -> str:
        candidate = candidate if kind == "verifier" else ""
        fields = self.key_fields(kind, question, context, choices, candidate)
        key = _sha(fields)
        cached = self._read(key, fields)
        if cached is not None:
            with self._lock:
                self.stats["hits"] += 1
            return cached
        while True:  # one request per key in flight; later callers wait for its result
            with self._lock:
                event = self._inflight.get(key)
                owner = event is None
                if owner:
                    event = self._inflight[key] = threading.Event()
            if owner:
                break
            event.wait()
            cached = self._read(key, fields)
            if cached is not None:
                return cached
            # The owner failed (maybe only because its own batch stopped): fetch under our own retries/stop.
        try:
            cached = self._read(key, fields)
            if cached is None:
                result = self._fetch(kind, question, context, choices, candidate, stop)
                cached = self._write(key, fields, result["output"],
                                     {"finish_reason": result["finish_reason"], "model": result["model"],
                                      **({"lora": self.served[kind]} if kind in self.served else {})})
                with self._lock:
                    self.stats["fetched"] += 1
                    self.stats["not_stopped"] += result["finish_reason"] != "stop"
            return cached
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()

    def call(
        self,
        agent_kind: str,
        example_id: int,
        question: str,
        context: str,
        choices: Dict[str, str],
        cache_namespace: str = "default",
        candidate_answer: str = "",
    ) -> str:
        text = self._get(agent_kind, question, context or "", choices, candidate_answer)
        with self._lock:
            self._call_log.append({"ts": int(time.time()), "agent_kind": agent_kind,
                                   "example_id": int(example_id), "output_len": len(text)})
        return text

    def prefetch(self, requests: Iterable[AdvisorRequest]) -> Dict[str, int]:
        """Fetch every uncached request concurrently. The first failure cancels this batch's queued
        requests, stops its running ones after their current attempt and raises at once (other
        batches, e.g. the collector's lookahead for earlier roots, are not affected)."""
        todo: Dict[str, AdvisorRequest] = {}
        n = 0
        for r in requests:
            n += 1
            candidate = r.candidate if r.kind == "verifier" else ""
            fields = self.key_fields(r.kind, r.question, r.context, r.choices, candidate)
            key = _sha(fields)
            if key not in todo and self._read(key, fields) is None:
                todo[key] = r
        if not todo:
            return {"requested": n, "fetched": 0}
        if not self._checked and self.server_url:
            self.check_server()
        stop = threading.Event()
        pool = ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(todo))))
        try:
            futures = [pool.submit(self._get, r.kind, r.question, r.context, r.choices, r.candidate, stop=stop)
                       for r in todo.values()]
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return {"requested": n, "fetched": len(todo)}

    # ------------------------------------------------- RemoteSubagentPool API
    def clear_cache(self) -> None:
        with self._lock:
            self._memory.clear()

    def drain_log(self) -> List[Dict[str, Any]]:
        with self._lock:
            log, self._call_log = self._call_log, []
        return log

