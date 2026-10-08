import json
import threading

import pytest

from mcq_rsi_helpers import FakeVLLM, fake_adapters, make_rows
from src.manager.mcq_rsi import advisors, prompts
from src.manager.mcq_rsi.advisors import (AdvisorError, AdvisorFailStop, AdvisorRequest, CachedAdvisorPool,
                                          FailStopEvalPool, eval_gate)

URL = "http://advisors:8000"


def pool(tmp_path, http=None, adapters=None, **kw):
    """A pool on ``tmp_path/cache``; with ``http`` it talks to the fake server serving the fake adapters."""
    sleeps = []
    if http is not None and adapters is None:
        cards, adapters = fake_adapters(tmp_path)
        http.served = cards
    p = CachedAdvisorPool("medqa", tmp_path / "cache", URL if http else None, http=http, adapters=adapters,
                          sleep=sleeps.append, **kw)
    return p, sleeps


def ask(p, kind, row, candidate="", eid=None, ns="default"):
    return p.call(kind, row["example_id"] if eid is None else eid, row["question"], row["context"], row["choices"],
                  cache_namespace=ns, candidate_answer=candidate)


def test_cache_key_semantics(tmp_path, monkeypatch):
    p, _ = pool(tmp_path)
    row = make_rows(1)[0]
    key = lambda kind, cand="", **over: p.key(kind, over.get("q", row["question"]), "", over.get("c", row["choices"]), cand)
    # The candidate is part of the key for the Verifier only.
    assert key("extractor", "A") == key("extractor", "B") == key("extractor")
    assert key("reasoner", "A") == key("reasoner")
    assert key("verifier", "A") != key("verifier", "B")
    assert len({key(k) for k in ("extractor", "reasoner", "verifier")}) == 3
    # Same stem, different options -> different key; question_hash is part of it.
    assert key("extractor", c={**row["choices"], "A": "other"}) != key("extractor")
    fields = p.key_fields("verifier", row["question"], "", row["choices"], "C")
    assert fields["candidate"] == "C" and fields["prompt_sha256"] == prompts.prompt_sha256("medqa", "verifier")
    assert fields["adapter"] == advisors.adapter_identity(p.spec, "verifier") and "#sha256=" in fields["adapter"]
    assert "candidate" not in p.key_fields("reasoner", row["question"], "", row["choices"], "C")
    # Prompt and adapter identity change the key.
    before = key("reasoner")
    monkeypatch.setitem(p.prompt_shas, "reasoner", "0" * 64)
    assert key("reasoner") != before
    p2 = CachedAdvisorPool("medqa", tmp_path / "cache", adapters={"reasoner": "hf://other#sha256=1"})
    assert p2.key("reasoner", row["question"], "", row["choices"]) not in (before, key("reasoner"))


def test_request_matches_paper_decoding_and_persists(tmp_path):
    http = FakeVLLM()
    p, _ = pool(tmp_path, http)
    row = make_rows(1)[0]
    out = ask(p, "verifier", row, "B", ns="marginal_value")
    assert out == out.strip() and http.gets == [f"{URL}/v1/models"]
    body = http.posts[0]
    assert body["model"] == "medqa_verifier" and body["temperature"] == 0.0 and body["max_tokens"] == 1024
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == prompts.build_advisor_messages("medqa", "verifier", row["question"], "", row["choices"],
                                                              candidate_answer="B")
    # cache_namespace and example_id are not part of the key; a second process reads the file.
    assert ask(p, "verifier", row, "B", eid=999, ns="eval") == out and len(http.posts) == 1
    fresh = CachedAdvisorPool("medqa", tmp_path / "cache", adapters=p.adapters)  # offline
    assert ask(fresh, "verifier", row, "B") == out
    with pytest.raises(AdvisorError, match="offline"):
        ask(fresh, "extractor", row)
    assert not list((tmp_path / "cache").rglob("*.part"))
    files = list((tmp_path / "cache").rglob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["output"] == out


def test_retry_then_success_and_fail_stop(tmp_path):
    row = make_rows(1)[0]
    http = FakeVLLM(fail=2)
    p, sleeps = pool(tmp_path, http)
    assert ask(p, "extractor", row).startswith("medqa_extractor")
    assert sleeps == [2.0, 4.0] and p.stats["retries"] == 2
    http = FakeVLLM(fail=100)
    p, sleeps = pool(tmp_path / "b", http, retries=3)
    with pytest.raises(AdvisorError, match="4 attempts"):
        ask(p, "reasoner", row)
    assert len(http.posts) == 4 and sleeps == [2.0, 4.0, 8.0]
    assert not list((tmp_path / "b").rglob("*.json"))  # nothing cached, no {"error": ...} text anywhere
    with pytest.raises(AdvisorError):
        p.prefetch([AdvisorRequest.for_row("reasoner", row)])


def test_identity_check_failure(tmp_path):
    row = make_rows(1)[0]
    cards, adapters = fake_adapters(tmp_path)
    renamed = [dict(c, id="mmlu_pro_verifier") if c["id"] == "medqa_verifier" else c for c in cards]
    http = FakeVLLM(renamed)
    p, _ = pool(tmp_path, http, adapters=adapters)
    with pytest.raises(AdvisorError, match="medqa_verifier"):
        p.check_server()
    with pytest.raises(AdvisorError, match="does not serve"):
        ask(p, "extractor", row)
    assert http.posts == []
    with pytest.raises(AdvisorError, match="server_url"):
        CachedAdvisorPool("medqa", tmp_path / "c2").check_server()


def test_identity_check_binds_served_weights_to_the_pinned_sha(tmp_path):
    """The right LoRA *names* are not enough: the served directory must hold the pinned weights on the base."""
    cards, adapters = fake_adapters(tmp_path)
    p, _ = pool(tmp_path, FakeVLLM(cards), adapters=adapters)
    assert p.check_server() and set(p.served) == {"extractor", "reasoner", "verifier"}
    # Retrained/other weights behind the same name.
    (tmp_path / "adapters" / "medqa" / "reasoner" / "adapter_model.safetensors").write_bytes(b"other weights")
    p, _ = pool(tmp_path, FakeVLLM(cards), adapters=adapters)
    with pytest.raises(AdvisorError, match="medqa_reasoner serves .* expected the pinned"):
        p.check_server()
    cards, adapters = fake_adapters(tmp_path / "fresh")
    # Names only (no root), a root this host cannot read, or a different base model.
    for broken in ({"root": None}, {"root": str(tmp_path / "missing")}, {"parent": "Qwen/Qwen3-4B"}):
        bad = [dict(c, **broken) if c["id"] == "medqa_verifier" else c for c in cards]
        p, _ = pool(tmp_path / "fresh", FakeVLLM(bad), adapters=adapters)
        with pytest.raises(AdvisorError, match="medqa_verifier"):
            p.check_server()
    # The registry identity (the default) pins the published adapter sha256: fake weights never pass.
    p = CachedAdvisorPool("medqa", tmp_path / "c3", URL, http=FakeVLLM(cards))
    assert p.adapters["verifier"] == advisors.adapter_identity(p.spec, "verifier")
    with pytest.raises(AdvisorError, match="expected the pinned"):
        p.check_server()


def test_prefetch_concurrent_dedup_and_put_conflict(tmp_path):
    rows = make_rows(6)
    http = FakeVLLM()
    p, _ = pool(tmp_path, http, workers=8)
    reqs = [AdvisorRequest.for_row(k, r, c) for r in rows for k in ("extractor", "reasoner", "verifier")
            for c in ("A", "B")]
    stats = p.prefetch(reqs + reqs)
    # extractor/reasoner ignore the candidate: 6 rows x (1 + 1 + 2) unique keys
    assert stats == {"requested": 72, "fetched": 24} and len(http.posts) == 24
    assert p.prefetch(reqs)["fetched"] == 0 and len(http.posts) == 24
    # Concurrent call() on one uncached key issues one request.
    http2 = FakeVLLM()
    p2, _ = pool(tmp_path / "x", http2)
    outs = []
    threads = [threading.Thread(target=lambda: outs.append(ask(p2, "reasoner", rows[0]))) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(outs)) == 1 and len(http2.posts) == 1
    # Seeding the cache (parity replay) refuses a conflicting output.
    key = p.put("extractor", rows[0]["question"], "", rows[0]["choices"], ask(p, "extractor", rows[0]))
    assert key == p.key("extractor", rows[0]["question"], "", rows[0]["choices"])
    with pytest.raises(AdvisorError, match="different"):
        p.put("extractor", rows[0]["question"], "", rows[0]["choices"], "something else")


def test_prefetch_failure_propagates(tmp_path):
    rows = make_rows(4)
    http = FakeVLLM(fail_when=lambda body: body["model"] == "medqa_verifier")
    p, _ = pool(tmp_path, http, retries=1)
    with pytest.raises(AdvisorError, match="medqa_verifier"):
        p.prefetch([AdvisorRequest.for_row(k, r, "A") for r in rows for k in ("extractor", "verifier")])


def test_prefetch_failure_returns_without_waiting_and_spares_other_batches(tmp_path):
    rows = make_rows(2)
    release = threading.Event()
    slow = rows[0]["question"]
    http = FakeVLLM(fail_when=lambda body: body["model"] == "medqa_verifier",
                    hold=lambda body: release if slow in body["messages"][-1]["content"] else None)
    p, _ = pool(tmp_path, http, retries=0, workers=4)
    # A slow request of another batch (e.g. the lookahead of an earlier root) is in flight.
    other = threading.Thread(target=lambda: p.prefetch([AdvisorRequest.for_row("extractor", rows[0])]))
    other.start()
    import time
    t0 = time.monotonic()
    with pytest.raises(AdvisorError, match="medqa_verifier"):
        p.prefetch([AdvisorRequest.for_row("verifier", rows[1], "A"), AdvisorRequest.for_row("extractor", rows[0])])
    assert time.monotonic() - t0 < 5  # did not wait for the held request
    release.set()
    other.join(10)
    # The other batch was not aborted by this batch's failure: its output is cached.
    assert ask(CachedAdvisorPool("medqa", tmp_path / "cache", adapters=p.adapters), "extractor", rows[0])
    # A pool-wide abort (collector failure) stops every later attempt until cleared.
    p.abort()
    with pytest.raises(AdvisorError, match="stopped"):
        ask(p, "reasoner", rows[1])
    p.clear_abort()
    assert ask(p, "reasoner", rows[1])


def test_lost_create_race_adopts_the_first_output(tmp_path):
    """Two pools on one cache directory (in one process) take the create-if-absent loser path deterministically;
    ``test_first_writer_wins_across_processes`` races real processes."""
    row = make_rows(1)[0]
    first, _ = pool(tmp_path, FakeVLLM(text=lambda body: "first output"))
    second, _ = pool(tmp_path, FakeVLLM(text=lambda body: "second output"), adapters=first.adapters)
    fields = second.key_fields("reasoner", row["question"], "", row["choices"])
    key = advisors._sha(fields)
    assert ask(first, "reasoner", row) == "first output"
    # The second pool fetched concurrently and lost the create-if-absent race: it adopts the first output.
    assert second._write(key, fields, "second output", {}) == "first output"
    assert ask(second, "reasoner", row) == "first output"
    assert json.loads(first._path(key).read_text())["output"] == "first output"
    assert not list((tmp_path / "cache").rglob("*.part"))


def _race_write(cache, adapters, fields, key, output, barrier, results):
    p = CachedAdvisorPool("medqa", cache, adapters=adapters)
    barrier.wait(30)
    results.put((output, p._write(key, fields, output, {"writer": output})))


def test_first_writer_wins_across_processes(tmp_path):
    """Separate processes racing on one uncached key all return the one stored output."""
    import multiprocessing
    row = make_rows(1)[0]
    _, adapters = fake_adapters(tmp_path)
    p = CachedAdvisorPool("medqa", tmp_path / "cache", adapters=adapters)
    fields = p.key_fields("reasoner", row["question"], "", row["choices"])
    key = advisors._sha(fields)
    ctx = multiprocessing.get_context("spawn")
    n = 6
    barrier, results = ctx.Barrier(n), ctx.Queue()
    procs = [ctx.Process(target=_race_write, args=(str(tmp_path / "cache"), adapters, fields, key, f"output {i}",
                                                   barrier, results)) for i in range(n)]
    [proc.start() for proc in procs]
    got = [results.get(timeout=60) for _ in procs]
    [proc.join(30) for proc in procs]
    assert all(proc.exitcode == 0 for proc in procs)
    stored = json.loads(p._path(key).read_text())
    assert {returned for _, returned in got} == {stored["output"]}
    winner = [mine for mine, returned in got if mine == returned]
    assert winner == [stored["output"]] and stored["meta"] == {"writer": stored["output"]}
    assert ask(p, "reasoner", row) == stored["output"]
    assert [f.name for f in p._path(key).parent.iterdir()] == [p._path(key).name]  # no .part left behind


def test_waiter_refetches_after_the_owning_request_fails(tmp_path):
    """A caller waiting on another caller's in-flight request for the same key does not inherit its failure:
    it fetches under its own retries once the owner gives up."""
    row = make_rows(1)[0]
    release, posted = threading.Event(), threading.Event()

    def hold(body):
        if len(http.posts) == 1:  # the owner's only attempt: hold it, then fail it
            posted.set()
            return release
        return None
    http = FakeVLLM(fail=1, hold=hold)
    p, _ = pool(tmp_path, http, retries=0)
    key = p.key("reasoner", row["question"], "", row["choices"])
    results = {}

    def run(name):
        try:
            results[name] = ask(p, "reasoner", row)
        except AdvisorError as e:
            results[name] = e
    owner = threading.Thread(target=run, args=("owner",))
    owner.start()
    assert posted.wait(10)
    waiting = threading.Event()
    real = p._inflight[key]

    class Spy:  # the in-flight marker the waiter blocks on; the owner sets the real one
        def wait(self, *a):
            waiting.set()
            return real.wait(*a)
    p._inflight[key] = Spy()
    waiter = threading.Thread(target=run, args=("waiter",))
    waiter.start()
    assert waiting.wait(10)  # the waiter is blocked on the owner's request
    release.set()
    owner.join(10)
    waiter.join(10)
    assert isinstance(results["owner"], AdvisorError)
    assert isinstance(results["waiter"], str) and results["waiter"].startswith("medqa_reasoner")
    assert len(http.posts) == 2 and p.stats["failed"] == 1 and p.stats["fetched"] == 1


def test_corrupt_or_empty_entries_are_refetched_never_trusted(tmp_path):
    row = make_rows(1)[0]
    http = FakeVLLM()
    p, _ = pool(tmp_path, http)
    out = ask(p, "extractor", row)
    path = p._path(p.key("extractor", row["question"], "", row["choices"]))
    for bad in (path.read_text()[:25], json.dumps({**json.loads(path.read_text()), "output": "  "})):
        path.write_text(bad)  # truncated entry / empty output from an older writer
        fresh, _ = pool(tmp_path, http, adapters=p.adapters)
        with pytest.warns(RuntimeWarning, match="refetching"):
            assert ask(fresh, "extractor", row) == out
        assert json.loads(path.read_text())["output"] == out and fresh.stats["corrupt"] == 1
    assert len(list(path.parent.glob(path.name + ".corrupt-*"))) == 2  # kept aside for inspection
    # Offline, a corrupt entry is a miss (fail-stop), not a JSONDecodeError.
    path.write_text("{")
    with pytest.warns(RuntimeWarning), pytest.raises(AdvisorError, match="offline"):
        ask(CachedAdvisorPool("medqa", tmp_path / "cache", adapters=p.adapters), "extractor", row)


def test_empty_advisor_output_is_an_error_and_never_cached(tmp_path):
    row = make_rows(1)[0]
    http = FakeVLLM(text=lambda body: "   ")
    p, sleeps = pool(tmp_path, http, retries=2)
    with pytest.raises(AdvisorError, match="no text content"):
        ask(p, "reasoner", row)
    assert len(http.posts) == 3 and p.stats["failed"] == 1
    assert not list((tmp_path / "cache").rglob("*.json"))
    with pytest.raises(AdvisorError, match="empty"):
        p.put("reasoner", row["question"], "", row["choices"], "")


def test_eval_gate_and_fail_stop_through_the_stages_eval_loop(tmp_path, monkeypatch):
    """``stages.run_eval_manager_tools`` swallows ``Exception`` from ``pool.call`` into ``{"error": ...}``.

    The raw pool therefore yields a report with malformed_tool_calls == 1, which ``eval_gate`` rejects;
    ``FailStopEvalPool`` aborts the eval instead, so no error payload ever reaches the manager.
    """
    import types

    import torch

    from mcq_rsi_helpers import tiny_tokenizer
    from src.benchmarks.base import StandardRow
    from src.pipeline import stages
    from src.subagents import runtime

    tok = tiny_tokenizer()
    outputs = ["DRAFT_ANSWER_A\n\n<tool_call>\n<function=extractor_tool>\n</function>\n</tool_call>",
               "DRAFT_ANSWER_A\nANSWER_A"]
    seen = []

    class EvalTok:  # the tiny tokenizer, decoding the scripted turns (Qwen3.5 keeps <tool_call> text)
        pad_token_id, eos_token_id = tok.pad_token_id, tok.eos_token_id

        def apply_chat_template(self, *a, **kw):
            return tok.apply_chat_template(*a, **kw)

        def __call__(self, text, **kw):
            seen.append(text)
            return tok(text, **kw)

        def decode(self, ids, **kw):
            return outputs[min(len(seen), 2) - 1]

    class EvalModel:
        def generate(self, input_ids, **kw):
            return torch.cat([input_ids, torch.tensor([[tok.eos_token_id]])], 1)

    row = make_rows(1)[0]
    rows = [StandardRow(**{k: row[k] for k in ("example_id", "benchmark_name", "task_subtype", "question",
                                               "context", "choices", "ground_truth")})]
    ctx = types.SimpleNamespace(seed=0, binding_mode="environment", teacher_id="t", eval_root=str(tmp_path / "eval"))
    (tmp_path / "mgr").mkdir()
    monkeypatch.setattr(stages, "_load_manager_for_eval", lambda *a: (EvalTok(), EvalModel()))
    failing = FakeVLLM(fail=100)
    raw, _ = pool(tmp_path, failing, retries=0)

    def run(p):
        seen.clear()
        monkeypatch.setattr(runtime, "RemoteSubagentPool", lambda server_url: p)
        return stages.run_eval_manager_tools(ctx, rows, manager_dir=str(tmp_path / "mgr"), n_samples=1,
                                             subagent_server_url=URL)

    report = run(raw)
    assert report["malformed_tool_calls"] == 1 and '{"error"' in seen[-1]  # the payload the manager saw
    assert eval_gate(report, raw) == ["malformed_tool_calls=1", "advisor failures=1"]
    guarded, _ = pool(tmp_path / "g", FakeVLLM(fail=100), retries=0)
    with pytest.raises(AdvisorFailStop, match="medqa_extractor"):
        run(FailStopEvalPool(guarded))
    assert len(seen) == 1  # stopped before any prompt containing the error was rendered
    ok, _ = pool(tmp_path / "ok", FakeVLLM())
    report = run(FailStopEvalPool(ok))
    assert report["malformed_tool_calls"] == 0 and eval_gate(report, ok) == []
