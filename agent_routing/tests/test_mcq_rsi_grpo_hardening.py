"""FA-GRPO hardening: each test kills one surviving mutant of the PR3 mutation run (grpo.py, __main__.py)."""
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest

from mcq_rsi_helpers import FakeManager, advisor_server, make_rows, tiny_model, tiny_tokenizer
from src.manager.marginal_value import _make_sft_rows
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import collect as C
from src.manager.mcq_rsi import grpo as G
from src.manager.mcq_rsi import protocol as P
from src.manager.mcq_rsi.advisors import CachedAdvisorPool

MEDQA = registry.get("medqa")
KINDS = ["extractor", "reasoner", "verifier"]


@pytest.fixture(autouse=True)
def _threads():
    import torch
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    from peft import LoraConfig, get_peft_model
    root = tmp_path_factory.mktemp("fa_grpo_hardening")
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    model.save_pretrained(root / "base")
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, init_lora_weights=False,
                                            target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"]))
    peft.save_pretrained(root / "sft")
    tok.save_pretrained(root / "sft")
    return root


def candidate_text(body):
    """Advisor output that depends strongly on the whole request (so on the Verifier's candidate)."""
    user = body["messages"][-1]["content"]
    digest = hashlib.sha256(user.encode()).hexdigest()
    return " ".join(f"{body['model']} audit {digest[i:i + 6]}" for i in range(0, 48, 6))


def make_pool(tmp_path, name="cache"):
    http, adapters = advisor_server(tmp_path, text=candidate_text)
    return CachedAdvisorPool("medqa", Path(tmp_path) / name, "http://x", adapters=adapters, http=http,
                             sleep=lambda s: None)


class RecordingPool:
    def __init__(self, pool):
        self.pool, self.calls = pool, []

    def call(self, **kw):
        self.calls.append(kw)
        return self.pool.call(**kw)

    def __getattr__(self, name):
        return getattr(self.pool, name)


@pytest.fixture(scope="module")
def anchors(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("anchors")
    rows = make_rows(4)
    e = [r["example_id"] for r in rows]
    script = {e[0]: {"root": ("A", "commit")}, e[1]: {"root": ("A", "verifier"), "rev": {("extractor",): "B"}},
              e[2]: {"root": ("C", "commit")}, e[3]: {"root": ("A", "commit"), "rev": {("verifier",): "D"}}}
    pool = make_pool(tmp, "anchor_cache")
    out = []
    for r in rows:
        out += _make_sft_rows(C.collect_question(r, MEDQA, FakeManager(script), pool, max_depth=1))
    return out


def config(ckpt, **kw):
    base = dict(base_model=str(ckpt / "base"), base_revision=None, steps=2, questions_per_step=2, guard_every=2,
                guard_probe_size=3, device="cpu", learning_rate=5e-3)
    return {**base, **kw}


def fresh(ckpt, **kw):
    import torch
    cfg = G.FAGRPOConfig.from_dict(config(ckpt, **kw))
    cfg.validate()
    tok, model = G.load_policy(cfg, ckpt / "sft")
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    return cfg, tok, model, params, torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


# (a) guard_probe scores the reference under the REFERENCE adapter --------------------------------------

def test_guard_probe_scores_the_reference_adapter_after_the_policy_moves(ckpt):
    import torch
    cfg, tok, model, params, _ = fresh(ckpt)
    scorer = G._Scorer(tok, MEDQA, KINDS)
    rows = make_rows(3)
    same = G.guard_probe(model, scorer, rows)
    assert same["kl_dec"] == pytest.approx(0.0, abs=1e-12) and same["kl_root"] == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        for p in params:
            p.add_(0.5 * torch.randn_like(p))
    moved = G.guard_probe(model, scorer, rows)
    assert moved["kl_dec"] > 1e-6 and moved["kl_root"] > 1e-6
    # The training-time guard measures the trained policy (not S_k against itself), so it sees the same drift.
    guard = G._guard(model, scorer, rows, {"call_rate": same["call_rate"]}, cfg, 1)
    assert guard["kl_dec"] == pytest.approx(moved["kl_dec"]) and guard["kl_root"] == pytest.approx(moved["kl_root"])
    assert guard["call_rate"] == moved["call_rate"]
    # Scoring the reference with the reference adapter itself gives zero again: the probe really switches adapters.
    ref_vs_ref = G.guard_probe(model, scorer, rows, adapter=G.REFERENCE, reference=G.REFERENCE)
    assert ref_vs_ref["kl_dec"] == pytest.approx(0.0, abs=1e-12)


# (b) _advise sends the current draft as the Verifier candidate ------------------------------------------

def test_verifier_receives_the_root_draft_and_its_output_shapes_J(ckpt, tmp_path):
    import torch
    cfg, tok, model, params, opt = fresh(ckpt, anchor_lambda=0.0)
    pool = RecordingPool(make_pool(tmp_path))
    scorer = G._Scorer(tok, MEDQA, KINDS)
    batch = make_rows(4)[1:3]
    report = G._train_step(1, model, tok, scorer, opt, params, batch, [], pool, cfg)
    drafts = {q["example_id"]: q["draft"] for q in report["questions"]}
    verifier = [c for c in pool.calls if c["agent_kind"] == "verifier"]
    assert len(verifier) == 2 and all(c["candidate_answer"] == drafts[c["example_id"]] for c in verifier)
    assert all(c["candidate_answer"] == "" for c in pool.calls if c["agent_kind"] != "verifier")
    G._use(model, G.REFERENCE)  # step 1 scored policy == S_k; the reference is unchanged by the update
    for row, q in zip(batch, report["questions"]):
        keys = list(row["choices"])
        gold = keys.index(row["ground_truth"])
        out = lambda cand: pool.pool.call(agent_kind="verifier", example_id=row["example_id"], question=row["question"],
                                          context="", choices=row["choices"], candidate_answer=cand)
        with torch.no_grad():
            J = lambda cand: float(scorer.key_logprobs(model, scorer.revision_prompt(row, "verifier", q["draft"], out(cand)),
                                                       keys)[0].exp()[gold])
            assert out(q["draft"]) != out("")
            assert q["J"]["verifier"] == pytest.approx(J(q["draft"]), abs=1e-6)
            assert abs(J(q["draft"]) - J("")) > 1e-7  # the candidate-free output is a different state


# (c) + (e) route-only anchor, tool block as a tuple in paper context ------------------------------------

def test_anchor_is_route_only_and_paper_context_renders_tools_as_tuple(anchors):
    from src.manager.routing_anchor import build_anchor_features
    tok = tiny_tokenizer()
    tools = P.TOOLS_SFT
    paper, _ = G.anchor_features(anchors, tok, G.FAGRPOConfig())
    evolve, _ = G.anchor_features(anchors, tok, G.FAGRPOConfig(anchor_context="evolve"))
    route_tuple, _ = build_anchor_features(list(anchors), tok, 4096, "route_only", tuple(tools))
    route_list, _ = build_anchor_features(list(anchors), tok, 4096, "route_only", list(tools))
    full_tuple, _ = build_anchor_features(list(anchors), tok, 4096, "full", tuple(tools))
    assert paper == route_tuple and evolve == route_list
    assert [f["input_ids"] for f in route_tuple] != [f["input_ids"] for f in route_list]  # the container matters
    assert [f["labels"] for f in route_tuple] != [f["labels"] for f in full_tuple]  # and so does the mode
    supervised = lambda fs: sum(sum(y != -100 for y in f["labels"]) for f in fs)
    assert 0 < supervised(paper) < supervised(full_tuple)


# (d) per-question loss normalisation: the step gradient is the mean of the per-question gradients -------

def test_step_gradient_is_the_mean_over_questions(ckpt, tmp_path, monkeypatch):
    import torch
    seen = []
    clip = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        params = list(params)
        seen.append([p.grad.detach().double().clone() for p in params])
        return clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    pool = make_pool(tmp_path)
    rows = make_rows(4)[1:3]
    for batch in ([rows[0]], [rows[1]], rows):
        cfg, tok, model, params, opt = fresh(ckpt, anchor_lambda=0.0, max_grad_norm=1e9)
        G._train_step(1, model, tok, G._Scorer(tok, MEDQA, KINDS), opt, params, batch, [], pool, cfg)
    g1, g2, both = seen
    norm = lambda gs: math.sqrt(sum(float((g ** 2).sum()) for g in gs))
    mean = [(a + b) / 2 for a, b in zip(g1, g2)]
    assert norm(both) > 0
    assert norm([x - y for x, y in zip(both, mean)]) <= 1e-5 * norm(mean)
    assert norm([x - (a + b) for x, a, b in zip(both, g1, g2)]) > 0.3 * norm(mean)  # not the sum


# (f) + (g) decRL tie tolerance is passed through; the tie-break is added after standardisation ----------

def test_question_objective_passes_the_tie_tolerance():
    import torch
    values, calls = [1.0, 0.3, 0.2, 0.995], [0, 1, 1, 1]
    cfg = G.FAGRPOConfig(decision_pg=True, decision_pg_epsilon=0.05, decision_pg_tie_tolerance=0.02,
                         beta_root=0.0, beta_dec=0.0)
    z = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    lsm = lambda x: torch.log_softmax(x, 0)
    k = torch.zeros(4, dtype=torch.float64)
    parts = G.question_objective(lsm(k), lsm(k), lsm(z), lsm(z.detach()), [], [], 0, cfg,
                                 decision_values=values, decision_calls=calls)
    got = torch.autograd.grad(parts["dec_pg"], z)[0]
    z2 = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    want = torch.autograd.grad(G.decision_pg_loss(lsm(z2), values, calls, 0.05, cfg.advantage_eps, 0.02), z2)[0]
    z3 = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    exact_ties = torch.autograd.grad(G.decision_pg_loss(lsm(z3), values, calls, 0.05, cfg.advantage_eps, 0.0), z3)[0]
    assert torch.allclose(got, want, atol=1e-12) and not torch.allclose(got, exact_ties, atol=1e-6)


def test_tie_break_is_added_after_standardisation():
    import torch
    values, calls, eps_tb, eps = [1.0, 1.0, 0.0, 0.4], [0, 1, 1, 1], 0.05, 1e-4
    z = torch.tensor([0.2, -0.1, 0.3, 0.0], dtype=torch.float64, requires_grad=True)
    grad = torch.autograd.grad(G.decision_pg_loss(torch.log_softmax(z, 0), values, calls, eps_tb, eps), z)[0]
    q = torch.softmax(z.detach(), 0)
    Q = torch.tensor(values, dtype=torch.float64)
    V = (q * Q).sum()
    sigma = torch.sqrt((q * (Q - V) ** 2).sum())
    A_after = (Q - V) / (sigma + eps) + torch.tensor(G.tie_break_bonus(values, calls, eps_tb), dtype=torch.float64)
    # loss = -sum_i sg[q_i A_i] log q_i  =>  dloss/dz = -(q*A - q * sum(q*A))
    expected = -(q * A_after - q * (q * A_after).sum())
    assert torch.allclose(grad, expected, atol=1e-12)
    Qb = Q + torch.tensor(G.tie_break_bonus(values, calls, eps_tb), dtype=torch.float64)  # bonus before standardising
    Vb = (q * Qb).sum()
    A_before = (Qb - Vb) / (torch.sqrt((q * (Qb - Vb) ** 2).sum()) + eps)
    assert not torch.allclose(expected, -(q * A_before - q * (q * A_before).sum()), atol=1e-6)


# (h) decRL action values: Q(commit) = 1[X = gold], Q(call a) = J_a ---------------------------------------

def test_decrl_action_values(ckpt, tmp_path, monkeypatch):
    seen = []
    real = G.question_objective

    def spy(*args, **kw):
        import inspect
        bound = inspect.signature(real).bind(*args, **kw).arguments
        out = real(*args, **kw)
        seen.append({"gold": bound["gold"], "values": list(bound["decision_values"]),
                     "calls": list(bound["decision_calls"]), "J": list(out["J"])})
        return out

    monkeypatch.setattr(G, "question_objective", spy)
    cfg, tok, model, params, opt = fresh(ckpt, decision_pg=True, anchor_lambda=0.0)
    import torch
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():  # policy != frozen reference, so Q(call a) must be J_a of the policy, not of S_k
        for p in params:
            p.add_(0.5 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    scorer = G._Scorer(tok, MEDQA, KINDS)
    rows = make_rows(8)
    report = G._train_step(1, model, tok, scorer, opt, params, rows, [], make_pool(tmp_path), cfg)
    commits = set()
    for row, q, s in zip(rows, report["questions"], seen):
        acts, _ = scorer.decision(row, q["draft"])
        assert [a.kind for a in acts] == [None, *KINDS] and s["calls"] == [0, 1, 1, 1]
        assert s["values"][0] == float(q["draft"] == row["ground_truth"])
        assert s["values"][1:] == pytest.approx(s["J"], abs=1e-12) and s["J"] == pytest.approx([q["J"][k] for k in KINDS])
        commits.add(s["values"][0])
    assert commits == {0.0, 1.0}  # both a correct and a wrong root draft were checked


# (i) source_sha256 is part of the GRPO run signature -----------------------------------------------------

def test_run_signature_pins_the_source_and_refuses_changed_code(ckpt, tmp_path, monkeypatch, anchors):
    pool = make_pool(tmp_path)
    rows = make_rows(4)
    cfg = G.FAGRPOConfig.from_dict(config(ckpt, steps=1))
    sig = G.run_signature(cfg, ckpt / "sft", rows, anchors, pool)
    assert sig["source_sha256"] == G.source_sha256() and len(sig["source_sha256"]) == 64
    out = tmp_path / "run"
    out.mkdir()
    (out / "training_run.json").write_text(json.dumps(sig, indent=2, sort_keys=True))
    monkeypatch.setattr(G, "source_sha256", lambda: "1" * 64)
    with pytest.raises(ValueError, match=r"inputs changed \(\['source_sha256'\]\)"):
        G.train_fa_grpo(config(ckpt, steps=1), ckpt / "sft", rows, anchors, pool, out)


# (j) load_policy loads a hub base at base_revision ---------------------------------------------------------

def test_load_policy_honours_base_revision_for_hub_ids(ckpt, monkeypatch):
    from src.subagents import train
    seen = []

    class Stop(Exception):
        pass

    def fake(model_id, **kw):
        seen.append((model_id, kw))
        raise Stop

    monkeypatch.setattr(train, "load_text_causal_model", fake)
    for base, revision, expected in (("Qwen/Qwen3.5-9B", "abc123", "abc123"), (str(ckpt / "base"), "abc123", None),
                                     ("Qwen/Qwen3.5-9B", None, None)):
        with pytest.raises(Stop):
            G.load_policy(G.FAGRPOConfig.from_dict(config(ckpt, base_model=base, base_revision=revision)), ckpt / "sft")
        model_id, kw = seen[-1]
        assert model_id == base and kw.get("revision") == expected


# (k) cmd_grpo defaults anchor_context to S_k's recorded SFT context --------------------------------------

def test_cli_grpo_defaults_anchor_context_to_the_recorded_sft_context(ckpt, tmp_path, monkeypatch, anchors):
    from src.manager.mcq_rsi import __main__ as cli
    from src.manager.mcq_rsi import splits
    stage = tmp_path / "sft_stage"
    shutil.copytree(ckpt / "sft", stage / "model")
    (stage / "sft_report.json").write_text(json.dumps({"config": {"context": "evolve"}}))
    anchor = tmp_path / "labels.jsonl"
    anchor.write_text("".join(json.dumps(r) + "\n" for r in anchors))
    captured = []

    def fake_train(config, checkpoint, rows, anchor_rows, pool, out, resume=True):
        captured.append(config)
        return {"steps": 0, "selected_step": 0, "accepted_by_guard": True,
                "informativeness": {"fraction": 0.0, "passed": False}, "final_dir": str(out)}

    monkeypatch.setattr(G, "train_fa_grpo", fake_train)
    monkeypatch.setattr(cli, "_make_pool", lambda args, bench: make_pool(tmp_path))
    monkeypatch.setattr(splits, "read_manifest", lambda path: {"benchmark": "medqa"})
    monkeypatch.setattr(splits, "pool_rows", lambda manifest, name: make_rows(4))
    argv = ["grpo", "--bench", "medqa", "--pool", "grpo_r2", "--checkpoint", str(stage / "model"), "--anchor", str(anchor),
            "--out", str(tmp_path / "g"), "--base-model", str(ckpt / "base"), "--base-revision", ""]
    assert cli.main(argv) == 0
    assert captured[-1]["anchor_context"] == "evolve"
    assert cli.main(argv + ["--anchor-context", "paper"]) == 0 and captured[-1]["anchor_context"] == "paper"
    assert cli.main([a if a != str(stage / "model") else str(ckpt / "sft") for a in argv]) == 0
    assert "anchor_context" not in captured[-1]  # no recorded context (round-1 S_1): the FAGRPOConfig default
