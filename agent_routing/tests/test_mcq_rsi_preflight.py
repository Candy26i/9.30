"""Advisor serving (renamed LoRA copies) and the preflight replay gate with a fake vLLM server (design §7.2)."""
import hashlib
import json
from pathlib import Path

import pytest

from mcq_rsi_helpers import FakeVLLM, read_fixture
from src.manager.marginal_value import ADVISOR_KINDS
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import preflight as PF
from src.manager.mcq_rsi import serving as S
from src.manager.mcq_rsi.advisors import AdvisorError, CachedAdvisorPool

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peft_adapter(path, seed=0):
    """A PEFT-layout LoRA (Qwen3_5ForCausalLM key names) with real tensors."""
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(2):
        for mod in ("self_attn.q_proj", "mlp.down_proj", "linear_attn.in_proj_qkv"):
            base = f"base_model.model.model.layers.{layer}.{mod}"
            tensors[f"{base}.lora_A.weight"] = torch.randn(2, 8, generator=g)
            tensors[f"{base}.lora_B.weight"] = torch.randn(8, 2, generator=g)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
    (path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, "peft_type": "LORA",
                                                          "base_model_name_or_path": registry.BASE_MODEL}))
    return tensors


# ------------------------------------------------------------------------------------ serving

def test_renamed_copy_holds_identical_tensors_under_the_multimodal_layout(tmp_path):
    from safetensors.torch import load_file
    import torch
    src = tmp_path / "src"
    original = peft_adapter(src)
    info = S.prepare_served_lora(src, tmp_path / "served", "multimodal", sha(src / "adapter_model.safetensors"))
    served = load_file(str(tmp_path / "served" / "adapter_model.safetensors"))
    assert info["n_tensors"] == info["renamed"] == len(original) == 12
    assert all(k.startswith(S.MULTIMODAL_PREFIX) for k in served)
    for k, v in original.items():
        assert torch.equal(served[k.replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")], v)
    assert S.key_layout(tmp_path / "served" / "adapter_model.safetensors") == {S.MULTIMODAL_PREFIX: 12}
    assert S.key_layout(src / "adapter_model.safetensors") == {S.TEXT_PREFIX: 12}
    assert S.data_sha256(src / "adapter_model.safetensors") == S.data_sha256(tmp_path / "served" / "adapter_model.safetensors")
    # vLLM's view: strip base_model.model., apply the Qwen3-VL mapper (model.language_model. -> language_model.model.).
    mapped = lambda k: k.replace("base_model.model.", "", 1).replace("model.language_model.", "language_model.model.", 1)
    assert {mapped(k).rsplit(".lora_", 1)[0] for k in served} == {f"language_model.model.layers.{i}.{m}" for i in range(2)
                                                                   for m in ("self_attn.q_proj", "mlp.down_proj",
                                                                             "linear_attn.in_proj_qkv")}
    assert all(mapped(k).startswith("model.layers.") for k in original)  # the original names map to no module
    S.verify_served_lora(tmp_path / "served", sha(src / "adapter_model.safetensors"))
    with pytest.raises(ValueError, match="pinned"):
        S.verify_served_lora(tmp_path / "served", "0" * 64)
    with pytest.raises(ValueError, match="pinned"):
        S.prepare_served_lora(src, tmp_path / "x", "multimodal", "0" * 64)
    assert S.prepare_served_lora(src, tmp_path / "y", "as_is")["served_root"] == str(src)


def test_tampered_or_foreign_copies_are_refused(tmp_path):
    src = tmp_path / "src"
    peft_adapter(src)
    pinned = sha(src / "adapter_model.safetensors")
    S.prepare_served_lora(src, tmp_path / "served", "multimodal", pinned)
    weights = tmp_path / "served" / "adapter_model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-3] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="provenance"):
        S.verify_served_lora(tmp_path / "served", pinned)
    # Re-recording the provenance does not help: the tensor bytes no longer equal the pinned file's.
    prov = json.loads((tmp_path / "served" / S.PROVENANCE).read_text())
    prov["served_sha256"] = sha(weights)
    (tmp_path / "served" / S.PROVENANCE).write_text(json.dumps(prov))
    with pytest.raises(ValueError, match="tensor data"):
        S.verify_served_lora(tmp_path / "served", pinned)
    with pytest.raises(ValueError, match="no mcq_rsi_served_lora"):
        S.verify_served_lora(src, pinned)
    with pytest.raises(ValueError, match="unexpected LoRA tensor name"):
        S.rename_key("base_model.model.lm_head.lora_A.weight")


def served_pool(tmp_path, mode="multimodal", tamper=False):
    """Fake server whose LoRA cards point at renamed copies of fake adapters (pinned by their source sha)."""
    cards, adapters = [], {}
    for i, kind in enumerate(ADVISOR_KINDS):
        src = tmp_path / "import" / kind
        peft_adapter(src, seed=i)
        pinned = sha(src / "adapter_model.safetensors")
        info = S.prepare_served_lora(src, tmp_path / "served" / f"medqa_{kind}", mode, pinned)
        cards.append({"id": f"medqa_{kind}", "root": info["served_root"], "parent": registry.BASE_MODEL})
        adapters[kind] = f"test://medqa/{kind}#sha256={'0' * 64 if tamper and kind == 'verifier' else pinned}"
    cards.append({"id": registry.BASE_MODEL, "root": registry.BASE_MODEL, "parent": None})
    return cards, adapters


def test_check_server_accepts_verified_renamed_copies_only(tmp_path):
    cards, adapters = served_pool(tmp_path)
    pool = CachedAdvisorPool("medqa", tmp_path / "cache", "http://x", adapters=adapters, http=FakeVLLM(cards))
    pool.check_server()
    assert {k: v["key_layout"] for k, v in pool.served.items()} == {k: "multimodal" for k in ADVISOR_KINDS}
    cards2, adapters2 = served_pool(tmp_path / "b", tamper=True)
    with pytest.raises(AdvisorError, match="medqa_verifier"):
        CachedAdvisorPool("medqa", tmp_path / "cache2", "http://x", adapters=adapters2, http=FakeVLLM(cards2)).check_server()
    cards3, adapters3 = served_pool(tmp_path / "c", mode="as_is")  # paper-era files served as-is: exact sha
    p3 = CachedAdvisorPool("medqa", tmp_path / "cache3", "http://x", adapters=adapters3, http=FakeVLLM(cards3))
    p3.check_server()
    assert all("key_layout" not in v for v in p3.served.values())


# ---------------------------------------------------------------------------------- preflight

def drifted(text):
    """``text`` diverging after 70% of its words (bf16 greedy drift on another GPU / other kernels)."""
    words = text.split()
    k = max(1, int(0.7 * len(words)))
    return " ".join(words[:k] + [f"drift{i}" for i in range(len(words) - k)])


def replay_server(tmp_path, mode):
    """Answers like a server under ``mode``: ok (LoRA applied, paper reproduced), trap (LoRA ignored: LoRA == base),
    paper_base (the recorded outputs are the base model's), drift (LoRA applied, outputs differ), near (LoRA
    applied, recorded outputs reproduced up to numeric drift), near_base (the base reproduces them up to drift)."""
    fx = read_fixture("paper_medqa_replay.json")
    cards, adapters = served_pool(tmp_path)
    pool = CachedAdvisorPool("medqa", tmp_path / "cache", "http://x", adapters=adapters, sleep=lambda s: None)
    rows = {int(r["example_id"]): r for r in fx["rows"]}
    recorded = {}
    for kind in ADVISOR_KINDS:
        for item in PF.recorded_items(fx["records"], rows, kind, 100):
            r = item["row"]
            body = pool.request_payload(kind, r["question"], r.get("context") or "", r["choices"], item["candidate"])
            recorded[json.dumps(body["messages"], sort_keys=True)] = item["output"].strip()

    def text(body):
        rec = recorded.get(json.dumps(body["messages"], sort_keys=True), "unrecorded")
        base = "base:" + hashlib.sha256(json.dumps(body["messages"]).encode()).hexdigest()[:16]
        if body["model"] == registry.BASE_MODEL:
            return rec if mode == "paper_base" else drifted(rec) if mode == "near_base" else base
        return {"ok": rec, "trap": base, "paper_base": "lora:" + base, "drift": "lora:" + base, "near": drifted(rec),
                "near_base": "lora:" + base}[mode]

    http = FakeVLLM(cards, text=text, version={"version": "0.26.0"})
    pool._http = http
    return pool, fx, http


@pytest.mark.parametrize("mode,passed,diagnosis", [("ok", True, None), ("trap", False, "LoRA not applied"),
                                                   ("paper_base", False, "plain base"), ("drift", False, "prompt registry"),
                                                   ("near", True, None), ("near_base", False, "plain base")])
def test_preflight_gate(tmp_path, mode, passed, diagnosis):
    pool, fx, http = replay_server(tmp_path, mode)
    result = PF.run_preflight("medqa", pool, fx["records"], fx["rows"], n_per_kind=5)
    assert result["passed"] is passed
    assert set(result["kinds"]) == set(ADVISOR_KINDS) and result["served_models"]
    assert result["vllm_version"] == {"version": "0.26.0"} and "gpu" in result and "server_flags" in result
    for kind, k in result["kinds"].items():
        assert 0 < k["n"] <= 5
        if mode == "ok":
            assert k["match_rate"] == 1.0 and k["lora_effect_rate"] == 1.0 and k["diagnosis"] is None
            assert k["replay"] == "exact" and k["median_lora_similarity"] == 1.0 and k["median_lora_prefix_ratio"] == 1.0
        elif passed:  # numeric drift: no exact match, but the LoRA is clearly closer to the recorded outputs
            assert k["lora_closer_rate"] == 1.0 and k["diagnosis"] is None and k["replay"] in ("exact", "closer_than_base")
            assert 0.5 <= k["median_lora_similarity"] <= 1.0 and k["median_base_similarity"] < 0.5
        else:
            assert diagnosis in k["diagnosis"] and not k["passed"] and k["replay"] == "failed"
    if mode == "near":  # long outputs really diverge (exact match fails) and still pass
        assert any(k["match_rate"] < 0.9 and k["replay"] == "closer_than_base" for k in result["kinds"].values())
    posts = [p for p in http.posts]
    # Exactly the cache's request: greedy, 1024 tokens, thinking off; Verifier with its recorded candidate.
    assert all(p["temperature"] == 0.0 and p["max_tokens"] == 1024 and p["chat_template_kwargs"] == {"enable_thinking": False}
               for p in posts)
    assert {p["model"] for p in posts} == {f"medqa_{k}" for k in ADVISOR_KINDS} | {registry.BASE_MODEL}
    verifier = [p for p in posts if p["model"] == "medqa_verifier"]
    assert verifier and all("CANDIDATE" in p["messages"][-1]["content"] or "candidate" in p["messages"][-1]["content"].lower()
                            for p in verifier)
    assert not list((tmp_path / "cache").rglob("*.json"))  # preflight never writes the advisor cache


def test_closer_than_base_needs_both_the_closer_rate_and_the_median_similarity():
    k = {"n": 20, "lora_effect_rate": 1.0, "match_rate": 0.2, "base_match_rate": 0.0,
         "lora_closer_rate": 0.8, "median_lora_similarity": 0.6, "base_closer_rate": 0.0, "median_base_similarity": 0.1}
    assert PF.diagnose(k, 0.9, 0.5) is None
    assert "not reproduced" in PF.diagnose({**k, "median_lora_similarity": 0.3}, 0.9, 0.5)  # closer, but far from both
    assert "not reproduced" in PF.diagnose({**k, "lora_closer_rate": 0.7}, 0.9, 0.5)
    assert PF.diagnose({**k, "match_rate": 0.95, "lora_closer_rate": 0.0}, 0.9, 0.5) is None
    assert "plain base" in PF.diagnose({**k, "lora_closer_rate": 0.1, "base_closer_rate": 0.9,
                                        "median_base_similarity": 0.7}, 0.9, 0.5)
    assert "LoRA not applied" in PF.diagnose({**k, "lora_effect_rate": 0.2}, 0.9, 0.5)


def test_recorded_items_carry_the_verifier_candidate():
    fx = read_fixture("paper_medqa_replay.json")
    rows = {int(r["example_id"]): r for r in fx["rows"]}
    items = PF.recorded_items(fx["records"], rows, "verifier", 50)
    assert items and all(i["candidate"] in "ABCD" and i["candidate"] for i in items)
    assert len({(i["example_id"], i["candidate"]) for i in items}) == len(items)
    assert all(i["candidate"] == "" for i in PF.recorded_items(fx["records"], rows, "extractor", 50))
    assert PF.recorded_items(fx["records"], rows, "reasoner", 2) == PF.recorded_items(fx["records"], rows, "reasoner", 2)


def test_controller_requires_a_passing_preflight_for_this_server(tmp_path, monkeypatch):
    from src.manager.mcq_rsi import advisors, controller
    pool, fx, http = replay_server(tmp_path, "ok")
    adapters = pool.adapters
    monkeypatch.setattr(advisors, "CachedAdvisorPool",
                        lambda bench, cache, url, workers=32: CachedAdvisorPool(bench, cache, url, adapters=adapters,
                                                                                http=http, workers=workers))
    cfg = controller.load_config({"bench": "medqa", "advisor_cache": str(tmp_path / "cache"),
                                  "import_dir": str(tmp_path / "import"), "advisor_url": "http://x"})
    with pytest.raises(RuntimeError, match="no advisor preflight"):
        controller.Runtime(cfg).pool()
    failed = PF.run_preflight("medqa", replay_server(tmp_path / "t", "trap")[0], fx["records"], fx["rows"], n_per_kind=2)
    PF.write_report(cfg["advisor_cache"], failed)
    with pytest.raises(RuntimeError, match="failed"):
        controller.Runtime(cfg).pool()
    PF.write_report(cfg["advisor_cache"], PF.run_preflight("medqa", pool, fx["records"], fx["rows"], n_per_kind=2))
    assert controller.Runtime(cfg).pool().served  # passes: same identity, server and served adapters
    good = PF.report_path(cfg["advisor_cache"], "medqa").read_text()
    other = json.loads(good)
    other["pool_identity"]["decode"] = {"temperature": 0.7}  # a report of another advisor identity
    PF.report_path(cfg["advisor_cache"], "medqa").write_text(json.dumps(other))
    with pytest.raises(RuntimeError, match="another advisor identity"):
        controller.Runtime(cfg).pool()
    other = json.loads(good)
    other["verified_loras"]["verifier"]["root"] = "/elsewhere"
    PF.report_path(cfg["advisor_cache"], "medqa").write_text(json.dumps(other))
    with pytest.raises(RuntimeError, match="other served adapters"):
        controller.Runtime(cfg).pool()
    assert controller.Runtime({**cfg, "preflight": {**cfg["preflight"], "required": False}}).pool()


def test_preflight_bound_to_vllm_version_and_server_flags(tmp_path, monkeypatch):
    from src.manager.mcq_rsi import advisors, controller
    pool, fx, http = replay_server(tmp_path, "ok")
    flags = tmp_path / "server_flags_18002.json"
    flags.write_text(json.dumps({"vllm_version": "0.26.0", "lora_mode": "multimodal", "args": ["--max-loras", "12"]}))
    monkeypatch.setenv("MCQ_ADVISOR_FLAGS_FILE", str(flags))
    monkeypatch.setattr(advisors, "CachedAdvisorPool",
                        lambda bench, cache, url, workers=32: CachedAdvisorPool(bench, cache, url, adapters=pool.adapters,
                                                                                http=http, workers=workers))
    cfg = controller.load_config({"bench": "medqa", "advisor_cache": str(tmp_path / "cache"),
                                  "import_dir": str(tmp_path / "import"), "advisor_url": "http://x"})
    rep = PF.run_preflight("medqa", pool, fx["records"], fx["rows"], n_per_kind=2)
    assert rep["server_flags"]["args"] == ["--max-loras", "12"]
    PF.write_report(cfg["advisor_cache"], rep)
    assert controller.Runtime(cfg).pool().served
    flags.write_text(json.dumps({"vllm_version": "0.26.0", "lora_mode": "multimodal", "args": ["--max-loras", "1"]}))
    with pytest.raises(RuntimeError, match="other flags"):
        controller.Runtime(cfg).pool()
    flags.write_text(json.dumps(rep["server_flags"]))
    http.version = {"version": "0.30.0"}
    with pytest.raises(RuntimeError, match="vLLM"):
        controller.Runtime(cfg).pool()
    http.version = {"version": "0.26.0"}
    assert controller.Runtime(cfg).pool().served
    # Without the env override the fingerprint is read next to the served LoRAs (server_flags_<port>.json).
    monkeypatch.delenv("MCQ_ADVISOR_FLAGS_FILE")
    pool2 = CachedAdvisorPool("medqa", tmp_path / "c2", "http://127.0.0.1:18002", adapters=pool.adapters, http=http)
    pool2.check_server()
    assert PF.server_flags(pool2) is None
    near = Path(next(iter(pool2.served.values()))["root"]).parent / "server_flags_18002.json"
    near.write_text(json.dumps({"args": ["x"]}))
    assert PF.server_flags(pool2) == {"args": ["x"]}


def test_sequential_recheck_compares_cached_batched_outputs_with_sequential_ones(tmp_path):
    from src.manager.mcq_rsi.advisors import AdvisorRequest
    from mcq_rsi_helpers import advisor_server, make_rows
    http, adapters = advisor_server(tmp_path)
    pool = CachedAdvisorPool("medqa", tmp_path / "cache", "http://x", adapters=adapters, http=http, workers=8)
    rows = make_rows(6)
    reqs = [AdvisorRequest.for_row(k, r) for r in rows for k in ("extractor", "reasoner")]
    reqs += [AdvisorRequest.for_row("verifier", r, "B") for r in rows]
    pool.prefetch(reqs)
    same = PF.sequential_recheck(pool, reqs, 4)
    assert same["workers"] == 8 and set(same["kinds"]) == set(ADVISOR_KINDS)
    assert all(k["n"] == 4 and k["exact_rate"] == 1.0 and k["median_similarity"] == 1.0 for k in same["kinds"].values())
    http.text = lambda body: "a different batched continuation"
    diff = PF.sequential_recheck(pool, reqs, 2)
    assert all(k["exact_rate"] == 0.0 and k["median_similarity"] < 1.0 for k in diff["kinds"].values())
    assert pool.cached(AdvisorRequest.for_row("verifier", rows[0], "C")) is None  # uncached: never fetched
    assert PF.sequential_recheck(CachedAdvisorPool("medqa", tmp_path / "cache", None, adapters=adapters), reqs, 2) == \
        {"skipped": "offline pool"}


def test_cli_preflight_exit_codes(tmp_path, monkeypatch, capsys):
    from src.manager.mcq_rsi import __main__ as cli
    from src.manager.mcq_rsi import advisors, splits
    for mode, code in (("ok", 0), ("trap", 1)):
        pool, fx, http = replay_server(tmp_path / mode, mode)
        imp = tmp_path / mode / "imp" / "medqa" / "round1"
        imp.mkdir(parents=True)
        (imp / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fx["records"]))
        monkeypatch.setattr(advisors, "CachedAdvisorPool",
                            lambda bench, cache, url, workers=32, a=pool.adapters, h=http:
                            CachedAdvisorPool(bench, cache, url, adapters=a, http=h))
        monkeypatch.setattr(splits, "read_manifest", lambda path: {"benchmark": "medqa"})
        monkeypatch.setattr(splits, "pool_rows", lambda manifest, name, rows=None, fx=fx:
                            fx["rows"] if name == "collect_r1" else pytest.fail(name))
        cfg = tmp_path / mode / "cfg.json"
        cfg.write_text(json.dumps({"bench": "medqa", "import_dir": str(tmp_path / mode / "imp"),
                                   "advisor_cache": str(tmp_path / mode / "cache")}))
        assert cli.main(["preflight", "--config", str(cfg), "--advisor-url", "http://x", "--n", "2"]) == code
        out = capsys.readouterr().out
        assert ("PASS" if code == 0 else "LoRA not applied") in out
        assert json.loads(PF.report_path(tmp_path / mode / "cache", "medqa").read_text())["passed"] is (code == 0)
        # --skip-if-passed keeps a passing report for this server (no replay); a failed one is replayed.
        posts = len(http.posts)
        assert cli.main(["preflight", "--config", str(cfg), "--advisor-url", "http://x", "--n", "2",
                         "--skip-if-passed"]) == code
        out = capsys.readouterr().out
        assert (len(http.posts) == posts and "skipped" in out) if code == 0 else (len(http.posts) > posts)
        if code == 0:  # other settings: replayed
            assert cli.main(["preflight", "--config", str(cfg), "--advisor-url", "http://x", "--n", "3",
                             "--skip-if-passed"]) == 0
            assert len(http.posts) > posts
