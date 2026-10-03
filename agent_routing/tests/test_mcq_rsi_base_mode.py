"""advisor_mode "base" (D14): every advisor is the base model with its role prompt, as the paper-era server
returned; pool requests and cache keys, the server identity check, the base-mode preflight gate, the config,
and the serving scripts."""
import hashlib
import json
from pathlib import Path

import pytest

from mcq_rsi_helpers import FakeVLLM, make_rows, read_fixture
from src.manager.marginal_value import ADVISOR_KINDS
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import controller as CT
from src.manager.mcq_rsi import preflight as PF
from src.manager.mcq_rsi.advisors import AdvisorError, CachedAdvisorPool, base_identity

ROOT = Path(__file__).resolve().parents[1]
BASE_CARD = {"id": registry.BASE_MODEL, "root": registry.BASE_MODEL, "parent": None}


def test_base_pool_requests_the_base_model_with_each_role_prompt_and_keys_its_own_cache(tmp_path):
    http = FakeVLLM([BASE_CARD])
    base = CachedAdvisorPool("medqa", tmp_path / "cache", "http://x", http=http, sleep=lambda s: None, mode="base")
    lora = CachedAdvisorPool("medqa", tmp_path / "cache", None)
    row = make_rows(1)[0]
    args = (row["question"], row.get("context") or "", row["choices"])
    for kind in ADVISOR_KINDS:
        cand = "A" if kind == "verifier" else ""
        pb, pl = base.request_payload(kind, *args, cand), lora.request_payload(kind, *args, cand)
        assert pb["model"] == registry.BASE_MODEL and pl["model"] == f"medqa_{kind}"
        assert pb["messages"] == pl["messages"]  # same role prompt and user turn
        assert base.key(kind, *args, cand) != lora.key(kind, *args, cand)
        assert base.key_fields(kind, *args, cand)["adapter"] == base_identity()
    ident = base.identity()
    assert ident["mode"] == "base" and set(ident["models"].values()) == {registry.BASE_MODEL} and "lora_names" not in ident
    assert "mode" not in lora.identity() and "lora_names" in lora.identity()  # the LoRA identity is unchanged
    out = base.call("verifier", int(row["example_id"]), *args, candidate_answer="B")
    assert http.posts[-1]["model"] == registry.BASE_MODEL and out.strip().startswith(registry.BASE_MODEL)
    assert base.served == {"base": {"model": registry.BASE_MODEL, "root": registry.BASE_MODEL}}
    assert base.call("verifier", int(row["example_id"]), *args, candidate_answer="B") == out
    assert len(http.posts) == 1  # cached
    with pytest.raises(ValueError, match="advisor mode"):
        CachedAdvisorPool("medqa", tmp_path / "c", None, mode="vllm")


def test_base_identity_check_needs_the_base_model_card(tmp_path):
    def pool(cards):
        return CachedAdvisorPool("medqa", tmp_path / "cache", "http://x", http=FakeVLLM(cards), mode="base")
    with pytest.raises(AdvisorError, match="does not serve the base model"):
        pool([{"id": "medqa_extractor", "parent": registry.BASE_MODEL, "root": "/a"}]).check_server()
    with pytest.raises(AdvisorError, match="does not serve the base model"):  # an adapter card under the base name
        pool([{"id": registry.BASE_MODEL, "parent": registry.BASE_MODEL, "root": "/a"}]).check_server()
    assert registry.BASE_MODEL in pool([BASE_CARD, {"id": "other"}]).check_server()


def base_replay_server(tmp_path, mode):
    """mode: paper (the base reproduces the recorded outputs), near (up to numeric drift), off (it does not)."""
    fx = read_fixture("paper_medqa_replay.json")
    pool = CachedAdvisorPool("medqa", tmp_path / "cache", "http://127.0.0.1:18002", sleep=lambda s: None, mode="base")
    rows = {int(r["example_id"]): r for r in fx["rows"]}
    recorded = {}
    for kind in ADVISOR_KINDS:
        for item in PF.recorded_items(fx["records"], rows, kind, 100):
            r = item["row"]
            body = pool.request_payload(kind, r["question"], r.get("context") or "", r["choices"], item["candidate"])
            recorded[json.dumps(body["messages"], sort_keys=True)] = item["output"].strip()

    def text(body):
        assert body["model"] == registry.BASE_MODEL
        rec = recorded[json.dumps(body["messages"], sort_keys=True)]
        if mode == "paper":
            return rec
        if mode == "near":
            words = rec.split()
            k = int(0.7 * len(words))
            return " ".join(words[:k] + ["drift"] * (len(words) - k))
        return "unrelated:" + hashlib.sha256(rec.encode()).hexdigest()

    pool._http = FakeVLLM([BASE_CARD], text=text, version={"version": "0.26.0"})
    return pool, fx


def _flags(tmp_path, monkeypatch, args):
    path = tmp_path / "server_flags_18002.json"
    path.write_text(json.dumps({"vllm_version": "0.26.0", "lora_mode": "none", "args": args}))
    monkeypatch.setenv("MCQ_SERVED_LORAS", str(tmp_path))
    return path


@pytest.mark.parametrize("mode,passed", [("paper", True), ("near", True), ("off", False)])
def test_base_preflight_replays_the_recorded_paper_outputs(tmp_path, monkeypatch, mode, passed):
    _flags(tmp_path, monkeypatch, ["--model", registry.BASE_MODEL, "--revision", registry.BASE_REVISION])
    pool, fx = base_replay_server(tmp_path, mode)
    result = PF.run_preflight("medqa", pool, fx["records"], fx["rows"], n_per_kind=5)
    assert result["advisor_mode"] == "base" and result["passed"] is passed
    assert {p["model"] for p in pool._http.posts} == {registry.BASE_MODEL}  # no LoRA is ever requested
    for k in result["kinds"].values():
        assert k["passed"] is passed and 0 < k["n"] <= 5
        if mode == "paper":
            assert k["match_rate"] == 1.0 and k["replay"] == "exact"
        elif mode == "near":
            assert k["replay"] == "similar" and k["median_base_similarity"] >= 0.5
        else:
            assert "does not reproduce" in k["diagnosis"]
    assert result["verified_loras"] == {"base": {"model": registry.BASE_MODEL, "root": registry.BASE_MODEL}}


def test_base_preflight_requires_the_pinned_revision_in_the_server_flags(tmp_path, monkeypatch):
    pool, fx = base_replay_server(tmp_path, "paper")
    monkeypatch.setenv("MCQ_SERVED_LORAS", str(tmp_path / "nowhere"))
    result = PF.run_preflight("medqa", pool, fx["records"], fx["rows"], n_per_kind=2)
    assert all(k["passed"] for k in result["kinds"].values()) and not result["passed"]
    assert "--revision" in result["flags_problem"]
    _flags(tmp_path, monkeypatch, ["--model", registry.BASE_MODEL, "--revision", "main"])
    assert not PF.run_preflight("medqa", pool, fx["records"], fx["rows"], n_per_kind=2)["passed"]
    _flags(tmp_path, monkeypatch, ["--revision", registry.BASE_REVISION])
    rep = PF.run_preflight("medqa", pool, fx["records"], fx["rows"], n_per_kind=2)
    assert rep["passed"] and rep["server_flags"]["lora_mode"] == "none"


def test_config_advisor_mode_is_validated_signed_and_reaches_the_pool(tmp_path, monkeypatch):
    raw = {"bench": "medqa", "import_dir": str(tmp_path / "i"), "advisor_cache": str(tmp_path / "c"),
           "advisor_url": None, "preflight": {"required": False}}
    assert CT.load_config(raw)["advisor_mode"] == "lora"
    cfg = CT.load_config({**raw, "advisor_mode": "base"})
    assert cfg["advisor_mode"] == "base" and CT.signed_config(cfg)["advisor_mode"] == "base"
    with pytest.raises(ValueError, match="advisor_mode"):
        CT.load_config({**raw, "advisor_mode": "vllm"})
    from src.manager.mcq_rsi import advisors
    seen = {}
    monkeypatch.setattr(advisors, "CachedAdvisorPool", lambda *a, **k: seen.update(k) or "pool")
    assert CT.Runtime(cfg).pool() == "pool" and seen["mode"] == "base"
    for b in ("medqa", "mmlu_pro", "gpqa", "aqua", "smoke"):
        assert json.loads((ROOT / "configs" / f"mcq_rsi_{b}.json").read_text())["advisor_mode"] == "base", b


def test_scripts_serve_the_base_model_only_for_base_advisors():
    start = (ROOT / "scripts" / "start_mcq_advisors.sh").read_text()
    assert 'LORAS="${MCQ_ADVISOR_LORAS:-all}"' in start and '[[ "$LORAS" == none ]]' in start
    assert 'if (( max_loras > 0 )); then' in start and '"$served_mode"' in start
    wrapper = (ROOT / "scripts" / "runpod_mcq_rsi.sh").read_text()
    assert 'MCQ_ADVISOR_LORAS="$loras"' in wrapper and '[[ "$(advisor_mode)" == base ]] && loras=none' in wrapper
