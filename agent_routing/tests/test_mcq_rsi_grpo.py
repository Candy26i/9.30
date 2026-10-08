"""FA-GRPO (design §3.5, §7.1 item 6): exact objectives, frozen reference, commits, guard, resume."""
import hashlib
import json
import math
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


@pytest.fixture(autouse=True)
def _threads():
    import torch
    torch.set_num_threads(1)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    """Tiny Qwen3.5 base + a non-trivial LoRA S_k (adapter + tokenizer + template)."""
    from peft import LoraConfig, get_peft_model
    root = tmp_path_factory.mktemp("fa_grpo")
    tok = tiny_tokenizer()
    model = tiny_model(tok)
    model.save_pretrained(root / "base")
    peft = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, init_lora_weights=False,
                                            target_modules=["q_proj", "v_proj", "gate_proj", "down_proj"]))
    peft.save_pretrained(root / "sft")
    tok.save_pretrained(root / "sft")
    return root


def make_pool(tmp_path, name="cache"):
    http, adapters = advisor_server(tmp_path)
    return CachedAdvisorPool("medqa", Path(tmp_path) / name, "http://x", adapters=adapters, http=http,
                             sleep=lambda s: None)


def anchor_rows(tmp_path):
    """Round SFT rows of every decision type, from the collector on a scripted manager."""
    rows = make_rows(4)
    e = [r["example_id"] for r in rows]
    script = {e[0]: {"root": ("A", "commit")}, e[1]: {"root": ("A", "verifier"), "rev": {("extractor",): "B"}},
              e[2]: {"root": ("C", "commit")}, e[3]: {"root": ("A", "commit"), "rev": {("verifier",): "D"}}}
    pool = make_pool(tmp_path, "anchor_cache")
    out = []
    for r in rows:
        out += _make_sft_rows(C.collect_question(r, MEDQA, FakeManager(script), pool, max_depth=1))
    assert {r["decision_type"] for r in out} == {"commit", "call", "commit_after_call"}
    return out


def config(ckpt, **kw):
    base = dict(base_model=str(ckpt / "base"), base_revision=None, steps=4, questions_per_step=2, guard_every=2,
                guard_probe_size=3, device="cpu", learning_rate=5e-3)
    return {**base, **kw}


def run(ckpt, tmp_path, out="run", rows=None, **kw):
    pool = make_pool(tmp_path, f"cache_{out}")
    return G.train_fa_grpo(config(ckpt, **kw), ckpt / "sft", rows or make_rows(8), anchor_rows(tmp_path), pool,
                           Path(tmp_path) / out)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- pure math

def test_exact_estimator_equals_monte_carlo_limit():
    import torch
    torch.manual_seed(0)
    for gold in (0, 3):
        z = torch.randn(5, dtype=torch.float64, requires_grad=True)
        loss, J = G.exact_pg_loss(torch.log_softmax(z, 0), gold)
        exact = torch.autograd.grad(loss, z)[0]
        # -grad L_pg = grad J / (sqrt(J(1-J)) + eps): the G -> inf limit of the group-normalised estimator.
        zj = z.detach().clone().requires_grad_(True)
        dJ = torch.autograd.grad(torch.softmax(zj, 0)[gold], zj)[0]
        assert torch.allclose(-exact, dJ / (math.sqrt(J * (1 - J)) + 1e-4), atol=1e-12)
        z2 = z.detach().clone().requires_grad_(True)
        sampled, J2 = G.sampled_pg_loss(torch.log_softmax(z2, 0), gold, 400_000,
                                        torch.Generator().manual_seed(1))
        mc = torch.autograd.grad(sampled, z2)[0]
        assert J2 == pytest.approx(J)
        assert torch.allclose(mc, exact, atol=1.5e-2), (mc, exact)
    # A deterministic state (J = 1) has zero advantage and zero gradient, as an all-equal GRPO group.
    z = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    loss, J = G.exact_pg_loss(torch.log_softmax(z, 0), 0)
    assert J == pytest.approx(1.0) and float(torch.autograd.grad(loss, z)[0].abs().max()) < 1e-12


def test_exact_kl_equals_brute_force():
    import torch
    p = torch.log_softmax(torch.tensor([0.3, -1.2, 2.0, 0.1], dtype=torch.float64), 0)
    q = torch.log_softmax(torch.tensor([0.0, 0.5, 1.0, -0.4], dtype=torch.float64), 0)
    brute = sum(math.exp(a) * (a - b) for a, b in zip(p.tolist(), q.tolist()))
    assert float(G.exact_kl(p, q)) == pytest.approx(brute, abs=1e-12)
    assert float(G.exact_kl(p, p)) == 0.0


def leaves(K=4, D=4, A=3):
    import torch
    torch.manual_seed(3)
    mk = lambda n: torch.randn(n, dtype=torch.float64, requires_grad=True)
    return mk(K), mk(D), [mk(K) for _ in range(A)]


def objective(cfg, root_z, dec_z, rev_z, gold=1, **kw):
    import torch
    lsm = lambda z: torch.log_softmax(z, 0)
    ref = lambda z: torch.log_softmax(z.detach() + 0.3 * torch.arange(len(z), dtype=z.dtype), 0)
    return G.question_objective(lsm(root_z), ref(root_z), lsm(dec_z), ref(dec_z), [lsm(z) for z in rev_z],
                                [ref(z) for z in rev_z], gold, cfg, **kw)


@pytest.mark.parametrize("estimator", ["exact", "sampled"])
def test_policy_gradient_never_reaches_root_or_decision_logits(estimator):
    import torch
    root_z, dec_z, rev_z = leaves()
    cfg = G.FAGRPOConfig(estimator=estimator)
    gens = [torch.Generator().manual_seed(i) for i in range(3)]
    parts = objective(cfg, root_z, dec_z, rev_z, generators=gens)
    g_root, g_dec, *g_rev = torch.autograd.grad(parts["pg"], [root_z, dec_z, *rev_z], allow_unused=True)
    assert g_root is None and g_dec is None
    assert any(g is not None and float(g.abs().max()) > 0 for g in g_rev)
    # The full loss reaches root/decision logits only through the KL terms: with beta_root = beta_dec = 0 not at all.
    cfg0 = G.FAGRPOConfig(estimator=estimator, beta_root=0.0, beta_dec=0.0)
    parts0 = objective(cfg0, root_z, dec_z, rev_z, generators=[torch.Generator().manual_seed(i) for i in range(3)])
    grads = torch.autograd.grad(parts0["loss"], [root_z, dec_z], allow_unused=True)
    assert all(g is None or float(g.abs().max()) == 0.0 for g in grads)


def test_decrl_ablation_adds_decision_gradient_only():
    import torch
    root_z, dec_z, rev_z = leaves()
    cfg = G.FAGRPOConfig(decision_pg=True, beta_root=0.0, beta_dec=0.0)
    parts = objective(cfg, root_z, dec_z, rev_z, decision_values=[0.0, 0.4, 0.7, 0.2], decision_calls=[0, 1, 1, 1])
    g_root, g_dec = torch.autograd.grad(parts["dec_pg"], [root_z, dec_z], allow_unused=True)
    assert g_root is None and float(g_dec.abs().max()) > 0
    # Exact action values: descent raises the best action (a call with Q = 0.7) and lowers the worst (commit, Q = 0).
    assert float(g_dec[2]) < 0 < float(g_dec[0])
    # Lexicographic variant breaks ties among argmax-Q actions towards fewer calls.
    z = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    loss = G.decision_pg_loss(torch.log_softmax(z, 0), [1.0, 1.0, 0.0], [0, 1, 1], epsilon=0.05)
    g = torch.autograd.grad(loss, z)[0]
    assert float(g[0]) < float(g[1])
    with pytest.raises(ValueError, match="MedQA only"):
        G.FAGRPOConfig(bench="gpqa", decision_pg=True).validate()


def test_informativeness_gate():
    assert G.informativeness([0.5, 0.99, 0.01, 0.98, 0.02])["n_informative"] == 1  # bounds are exclusive
    gate = G.informativeness([0.5] * 15 + [1.0] * 85)
    assert gate["fraction"] == pytest.approx(0.15) and gate["passed"]
    assert not G.informativeness([0.5] * 14 + [1.0] * 86)["passed"]
    assert not G.informativeness([])["passed"]


def test_select_rollback_picks_last_passing_guarded_step():
    g = lambda step, passed, enforced=True: {"step": step, "passed": passed, "enforced": enforced}
    assert G.select_rollback([g(8, True), g(16, True), g(24, False), g(32, True)]) == (16, 24)
    assert G.select_rollback([g(8, False)]) == (0, 8)
    assert G.select_rollback([g(8, True), g(16, True)]) == (16, None)
    assert G.select_rollback([g(8, False, enforced=False), g(16, True)]) == (16, None)  # decRL: logged only
    assert G.select_rollback([]) == (0, None)


# ------------------------------------------------------------------------- tiny model scoring

@pytest.fixture(scope="module")
def tiny():
    tok = tiny_tokenizer()
    return tok, tiny_model(tok)


def test_scoring_equals_brute_force_restricted_softmax(tiny):
    import torch
    tok, model = tiny
    row = make_rows(1)[0]
    keys = list(row["choices"])
    scorer = G._Scorer(tok, MEDQA, ["extractor", "reasoner", "verifier"])
    messages, prompt = scorer.root(row)
    assert prompt == P.HFBackend(model, tok).prompt_ids(messages, P.TOOLS_DEPLOY)

    def nxt(prefix):
        with torch.no_grad():
            return model(input_ids=torch.tensor([list(prompt) + list(prefix)])).logits[0, -1].double()

    with torch.no_grad():
        root_lp, gi = scorer.key_logprobs(model, prompt, keys)  # one-row fast path
        paths, trie = scorer.drafts(keys)
        general = P.score_paths(model, prompt, paths, trie, scorer.pad_id)
    assert torch.allclose(root_lp, general, atol=1e-5)
    key_tokens = [p[-1] for p in paths]
    row_logits = nxt(paths[0][:-1])
    brute = [float(row_logits[t] - torch.logsumexp(row_logits[key_tokens], 0)) for t in key_tokens]
    assert root_lp.tolist() == pytest.approx(brute, abs=1e-5)
    assert gi == max(range(len(keys)), key=lambda i: brute[i])
    # Decision: <= 4 legal turns after DRAFT_ANSWER_X, renormalised over the legal set; exact KL vs brute force.
    acts, dtrie = scorer.decision(row, keys[gi])
    assert [a.name for a in acts] == ["commit", "extractor", "reasoner", "verifier"]
    with torch.no_grad():
        dec_lp, di = G.score_tree(model, prompt, [a.ids for a in acts], dtrie, scorer.pad_id)
    brute_dec = []
    for a in acts:
        total = 0.0
        for t, token in enumerate(a.ids):
            allowed = dtrie.allowed(a.ids[:t])
            r = nxt(a.ids[:t])
            total += float(r[token] - torch.logsumexp(r[allowed], 0))
        brute_dec.append(total)
    assert dec_lp.tolist() == pytest.approx(brute_dec, abs=1e-4)
    assert float(torch.logsumexp(dec_lp, 0)) == pytest.approx(0.0, abs=1e-5)
    # The trie-greedy walk is the constrained greedy decision of the deployment manager.
    walk = P.constrained_greedy_walk(lambda prefix: nxt(prefix), dtrie)
    assert tuple(walk["path"]) == tuple(acts[di].ids)
    q = [x - 0.2 * i for i, x in enumerate(brute_dec)]
    qn = [x - math.log(sum(math.exp(y) for y in q)) for x in q]
    expected = sum(math.exp(a) * (a - b) for a, b in zip(brute_dec, qn))
    assert float(G.exact_kl(dec_lp.double(), torch.tensor(qn, dtype=torch.float64))) == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------- training runs

def test_end_to_end_tiny_run(ckpt, tmp_path):
    from peft import PeftConfig
    s = run(ckpt, tmp_path, steps=3)
    assert s["steps"] == 3 and s["accepted_by_guard"] and s["selected_step"] == 3 and s["rollback_step"] is None
    assert [g["step"] for g in s["guards"]] == [2, 3]  # every guard_every steps and at the last step
    assert s["informativeness"]["passed"] and s["informativeness"]["n_states"] == 3 * 2 * 3
    lines = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines] == [1, 2, 3]
    for r in lines:
        assert r["anchor_ce"] > 0 and r["anchor_supervised_tokens"] > 0 and math.isfinite(r["loss"])
        assert len(r["questions"]) == 2 and all(set(q["J"]) == {"extractor", "reasoner", "verifier"} for q in r["questions"])
    final = tmp_path / "run" / "final"
    assert sha(final / "adapter_model.safetensors") == s["final_adapter_sha256"] != sha(ckpt / "sft" / "adapter_model.safetensors")
    cfg, ref = PeftConfig.from_pretrained(final), PeftConfig.from_pretrained(ckpt / "sft")
    assert (cfg.r, cfg.lora_alpha, set(cfg.target_modules)) == (ref.r, ref.lora_alpha, set(ref.target_modules))
    assert (final / "chat_template.jinja").exists() and not (final / "optimizer.pt").exists()
    assert (final / "tokenizer_config.json").read_bytes() == (ckpt / "sft" / "tokenizer_config.json").read_bytes()
    # Disk: no tokenizer per step; only the selected (= newest) step keeps adapter weights, no optimizer after the end.
    steps = sorted((tmp_path / "run").glob("step-*"))
    assert len(steps) == 3 and not list((tmp_path / "run").glob("incomplete-*"))
    assert all(sorted(p.name for p in d.iterdir()) == ["README.md", "adapter_config.json", "step.json"]
               for d in steps[:2])
    assert sorted(p.name for p in steps[2].iterdir()) == ["README.md", "adapter_config.json",
                                                         "adapter_model.safetensors", "step.json"]
    # The summary's informativeness gate counts exactly the per-step counts (unrounded J in step.json).
    assert s["informativeness"]["n_informative"] == sum(r["n_informative"] for r in lines)
    # The Verifier was asked about the greedy root draft X of each training question.
    cache = [json.loads(p.read_text()) for p in (tmp_path / "cache_run").rglob("*.json")]
    drafts = {(q["example_id"], q["draft"]) for r in lines for q in r["questions"]}
    verifier = {e["fields"]["candidate"] for e in cache if e["fields"]["kind"] == "verifier"}
    assert {d for _, d in drafts} <= verifier
    # Rerun returns the stored summary; changed inputs or a foreign directory are refused.
    assert run(ckpt, tmp_path, steps=3) == s
    with pytest.raises(ValueError, match="inputs changed"):
        run(ckpt, tmp_path, steps=4)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "x").write_text("x")
    with pytest.raises(ValueError, match="without a matching"):
        run(ckpt, tmp_path, out="other")


def test_sampled_estimator_and_decrl_ablation_run(ckpt, tmp_path):
    s = run(ckpt, tmp_path, out="sampled", steps=2, estimator="sampled", num_generations=4)
    assert s["steps"] == 2 and all(math.isfinite(r["pg_loss"]) for r in s["per_step"])
    d = run(ckpt, tmp_path, out="decrl", steps=2, decision_pg=True, decision_pg_epsilon=0.05,
            guard_max_kl_dec=-1.0)  # the guard would fail, but decRL only logs it
    assert d["accepted_by_guard"] and d["guard_enforced"] is False and d["selected_step"] == 2
    assert all(not g["passed"] and not g["enforced"] for g in d["guards"])
    assert all("dec_pg_loss" in r and "dec_tiebreak_states" in r for r in d["per_step"])


def test_reference_adapter_is_frozen_copy_of_sft(ckpt, tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file
    sft = load_file(str(ckpt / "sft" / "adapter_model.safetensors"))
    seen = []
    commit = G._commit_step

    def spy(root, step, model, optimizer, report):
        ref = {n: p for n, p in model.named_parameters() if ".reference." in n}
        pol = {n: p for n, p in model.named_parameters() if ".default." in n}
        assert ref and not any(p.requires_grad for p in ref.values())
        assert all(p.requires_grad for p in pol.values())
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert opt_ids == {id(p) for p in pol.values()}
        diffs = []
        for key, value in sft.items():
            name = key.replace(".lora_A.weight", ".lora_A.reference.weight").replace(".lora_B.weight", ".lora_B.reference.weight")
            diffs.append(float((ref[name].detach().cpu() - value).abs().max()))
        moved = max(float((pol[n.replace(".reference.", ".default.")] - p).abs().max()) for n, p in ref.items())
        seen.append((step, max(diffs), moved))
        return commit(root, step, model, optimizer, report)

    monkeypatch.setattr(G, "_commit_step", spy)
    s = run(ckpt, tmp_path, steps=2)
    assert [x[0] for x in seen] == [1, 2]
    assert all(diff == 0.0 for _, diff, _ in seen)  # reference never changes
    assert all(moved > 0 for _, _, moved in seen)  # while the policy does
    first = s["per_step"][0]  # step 1 is scored before any update: policy == reference
    assert first["kl_root"] < 1e-9 and first["kl_dec"] < 1e-9 and first["kl_rev"] < 1e-9
    # After one update (lr > 0) the policy has moved away from the frozen reference: every KL is positive.
    # (Scoring the reference terms under the policy adapter would keep them at exactly 0.)
    second = s["per_step"][1]
    assert second["kl_root"] > 1e-6 and second["kl_dec"] > 1e-6 and second["kl_rev"] > 1e-6
    step2 = json.loads((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[1])
    assert all(q["kl_root"] > 0 and q["kl_dec"] > 0 and q["kl_rev"] > 0 for q in step2["questions"])
    assert s["baseline"]["call_rate"] == s["guards"][0]["baseline_call_rate"]
    with torch.no_grad():
        tok, model = G.load_policy(G.FAGRPOConfig.from_dict(config(ckpt)), ckpt / "sft", tmp_path / "run" / "final")
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(".default." in n for n in trainable)


class Crash(Exception):
    pass


def test_crash_between_commit_and_pointer_and_resume_determinism(ckpt, tmp_path, monkeypatch):
    from src.verifiable.rsi_grpo import committed_step_directories
    clean = run(ckpt, tmp_path, out="clean", steps=4)
    advance = G._advance_pointer

    def crash_at_3(root, names, step):
        if step == 3:
            raise Crash("killed after the step directory was renamed, before the pointer")
        return advance(root, names, step)

    monkeypatch.setattr(G, "_advance_pointer", crash_at_3)
    with pytest.raises(Crash):
        run(ckpt, tmp_path, out="crash", steps=4)
    out = tmp_path / "crash"
    assert len(list(out.glob("step-00003-*"))) == 1 and len(committed_step_directories(out)) == 2
    assert [json.loads(x)["step"] for x in (out / "metrics.jsonl").read_text().splitlines()] == [1, 2]
    monkeypatch.setattr(G, "_advance_pointer", advance)
    resumed = run(ckpt, tmp_path, out="crash", steps=4)
    chain = committed_step_directories(out)
    assert len(chain) == 4 and list(out.glob("step-00003-*")) == [chain[2]]  # the orphan is removed, never counted
    assert [json.loads(x)["step"] for x in (out / "metrics.jsonl").read_text().splitlines()] == [1, 2, 3, 4]
    assert [r["step"] for r in resumed["per_step"]] == [1, 2, 3, 4]
    assert resumed["informativeness"] == clean["informativeness"]
    # Bitwise-identical training: same final adapter and same per-step losses as the uninterrupted run.
    assert resumed["final_adapter_sha256"] == clean["final_adapter_sha256"]
    assert [r["loss"] for r in resumed["per_step"]] == [r["loss"] for r in clean["per_step"]]

    # A kill inside a step (before its commit) resumes identically too.
    train_step = G._train_step

    def die_in_step_2(step, *a, **kw):
        if step == 2:
            raise Crash("killed mid-step")
        return train_step(step, *a, **kw)

    monkeypatch.setattr(G, "_train_step", die_in_step_2)
    with pytest.raises(Crash):
        run(ckpt, tmp_path, out="mid", steps=4)
    monkeypatch.setattr(G, "_train_step", train_step)
    assert run(ckpt, tmp_path, out="mid", steps=4)["final_adapter_sha256"] == clean["final_adapter_sha256"]


def test_guard_failure_rolls_back_to_last_passing_step_and_ends(ckpt, tmp_path, monkeypatch):
    real = G._guard

    def scripted(model, scorer, probe, baseline, cfg, step):
        out = real(model, scorer, probe, baseline, cfg, step)
        return {**out, "passed": step != 4, "reasons": [] if step != 4 else ["scripted failure"]}

    monkeypatch.setattr(G, "_guard", scripted)
    s = run(ckpt, tmp_path, steps=6)
    out = tmp_path / "run"
    assert s["steps"] == 4 and not s["accepted_by_guard"] and s["guard_failed_step"] == 4
    assert s["rollback_step"] == 2 == s["selected_step"]
    step2 = next(out.glob("step-00002-*"))
    assert s["final_adapter_sha256"] == sha(step2 / "adapter_model.safetensors")
    assert not list(out.glob("step-00005-*"))
    # The stage has ended: re-finalising never trains again.
    (out / "summary.json").unlink()
    monkeypatch.setattr(G, "_train_step", lambda *a, **k: pytest.fail("trained after a guard failure"))
    again = run(ckpt, tmp_path, steps=6)
    assert again["selected_step"] == 2 and again["final_adapter_sha256"] == s["final_adapter_sha256"]


def test_first_guard_failure_returns_sft_checkpoint(ckpt, tmp_path):
    s = run(ckpt, tmp_path, steps=4, guard_max_kl_dec=-1.0)  # any KL fails the enforced guard
    assert s["steps"] == 2 and s["guard_failed_step"] == 2 and s["selected_step"] == 0 == s["rollback_step"]
    assert s["final_adapter_sha256"] == sha(ckpt / "sft" / "adapter_model.safetensors")
    assert "KL_dec" in s["guards"][0]["reasons"][0]


def test_cli_grpo(ckpt, tmp_path, monkeypatch):
    from src.manager.mcq_rsi import __main__ as cli
    from src.manager.mcq_rsi import splits
    rows = make_rows(8)
    pool = make_pool(tmp_path)
    anchor = tmp_path / "labels.jsonl"
    anchor.write_text("".join(json.dumps(r) + "\n" for r in anchor_rows(tmp_path)))
    (tmp_path / "cfg.json").write_text(json.dumps({"questions_per_step": 2, "guard_every": 1, "guard_probe_size": 2,
                                                   "device": "cpu"}))
    monkeypatch.setattr(cli, "_make_pool", lambda args, bench: pool)
    monkeypatch.setattr(splits, "read_manifest", lambda path: {"benchmark": "medqa"})
    monkeypatch.setattr(splits, "pool_rows", lambda manifest, name: rows if name == "grpo_r1" else [])
    argv = ["grpo", "--bench", "medqa", "--pool", "grpo_r1", "--checkpoint", str(ckpt / "sft"), "--anchor", str(anchor),
            "--out", str(tmp_path / "g"), "--config", str(tmp_path / "cfg.json"), "--steps", "2",
            "--base-model", str(ckpt / "base"), "--base-revision", ""]
    assert cli.main(argv) == 0
    s = json.loads((tmp_path / "g" / "summary.json").read_text())
    assert s["steps"] == 2 and [g["step"] for g in s["guards"]] == [1, 2] and s["config"]["learning_rate"] == 5e-6
    assert cli.main(argv) == 0  # complete: returns the stored summary
    assert s["config"]["anchor_context"] == "paper"
    argv2 = [a if a != str(tmp_path / "g") else str(tmp_path / "g2") for a in argv] + ["--anchor-context", "evolve"]
    assert cli.main(argv2[:argv2.index("--steps") + 1] + ["1"] + argv2[argv2.index("--steps") + 2:]) == 0
    assert json.loads((tmp_path / "g2" / "summary.json").read_text())["config"]["anchor_context"] == "evolve"


# ------------------------------------------------------------------- review fixes (PR3)

def test_bf16_base_reference_is_exact_sft_and_step1_kl_is_zero(ckpt, tmp_path, monkeypatch):
    """On a bf16 base (every CUDA run) peft's ``load_adapter`` would round the reference to bf16 before upcasting."""
    import torch
    from peft import PeftModel
    from peft.utils import get_peft_model_state_dict, load_peft_weights
    from src.subagents.train import load_text_causal_model
    saved = load_peft_weights(str(ckpt / "sft"), device="cpu")
    assert {v.dtype for v in saved.values()} == {torch.float32}
    # The hazard: plain peft loading of the second adapter on a bf16 base is not S_k.
    base = load_text_causal_model(str(ckpt / "base"), dtype=torch.bfloat16, trust_remote_code=True)
    plain = PeftModel.from_pretrained(base, str(ckpt / "sft"), adapter_name="default")
    plain.load_adapter(str(ckpt / "sft"), adapter_name="reference")
    rounded = get_peft_model_state_dict(plain, adapter_name="reference")
    assert any(not torch.equal(rounded[k].float(), v) for k, v in saved.items())
    del plain, base
    # load_policy reloads it: policy and reference are bitwise the fp32 S_k on the bf16 base.
    monkeypatch.setattr(G, "_dtype", lambda device: torch.bfloat16)
    tok, model = G.load_policy(G.FAGRPOConfig.from_dict(config(ckpt)), ckpt / "sft")
    assert {p.dtype for n, p in model.named_parameters() if "lora_" not in n} == {torch.bfloat16}
    for adapter in (G.POLICY, G.REFERENCE):
        state = get_peft_model_state_dict(model, adapter_name=adapter)
        assert all(state[k].dtype == torch.float32 and torch.equal(state[k].cpu(), v) for k, v in saved.items())
    del model
    s = run(ckpt, tmp_path, steps=1, guard_every=1)
    first = s["per_step"][0]
    assert first["kl_root"] == 0.0 and first["kl_dec"] == 0.0 and first["kl_rev"] == 0.0


def test_empty_anchor_with_positive_lambda_is_refused(ckpt, tmp_path):
    pool = make_pool(tmp_path)
    with pytest.raises(ValueError, match="no anchor rows"):
        G.train_fa_grpo(config(ckpt), ckpt / "sft", make_rows(8), [], pool, tmp_path / "x")
    assert not (tmp_path / "x").exists() or not any((tmp_path / "x").iterdir())
    tok = tiny_tokenizer()
    with pytest.raises(ValueError, match="no anchor rows"):
        G.anchor_features([], tok, G.FAGRPOConfig())
    assert G.anchor_features([], tok, G.FAGRPOConfig(anchor_lambda=0.0)) == ([], {})


def test_anchor_context_must_match_the_sft_context(ckpt, tmp_path):
    import shutil
    stage = tmp_path / "sft_stage"
    shutil.copytree(ckpt / "sft", stage / "model")
    (stage / "sft_report.json").write_text(json.dumps({"config": {"context": "evolve"}}))
    assert G.recorded_sft_context(stage / "model") == "evolve" and G.recorded_sft_context(ckpt / "sft") is None
    pool = make_pool(tmp_path)
    with pytest.raises(ValueError, match="anchor_context"):
        G.train_fa_grpo(config(ckpt), stage / "model", make_rows(8), anchor_rows(tmp_path), pool, tmp_path / "x")
    G.check_anchor_context(G.FAGRPOConfig(anchor_context="evolve"), stage / "model")
    G.check_anchor_context(G.FAGRPOConfig(anchor_lambda=0.0), stage / "model")  # no anchor, nothing to match


def test_decrl_tie_break_fires_on_near_ties_and_is_logged():
    import torch
    # X = gold (Q(commit) = 1) and a saturated-but-not-exact J_V = 0.995: no exact tie.
    values, calls = [1.0, 0.3, 0.2, 0.995], [0, 1, 1, 1]
    assert not any(G.tie_break_bonus(values, calls, 0.05, tolerance=0.0))
    bonus = G.tie_break_bonus(values, calls, 0.05, tolerance=0.02)
    assert bonus == pytest.approx([0.025, 0.0, 0.0, -0.025])
    z = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    with_tol = torch.autograd.grad(G.decision_pg_loss(torch.log_softmax(z, 0), values, calls, 0.05, 1e-4, 0.02), z)[0]
    z2 = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    exact = torch.autograd.grad(G.decision_pg_loss(torch.log_softmax(z2, 0), values, calls, 0.05, 1e-4, 0.0), z2)[0]
    assert float(with_tol[0]) < float(exact[0]) and float(with_tol[3]) > float(exact[3])
    root_z, dec_z, rev_z = leaves()
    cfg = G.FAGRPOConfig(decision_pg=True, decision_pg_epsilon=0.05)
    assert objective(cfg, root_z, dec_z, rev_z, decision_values=values, decision_calls=calls)["dec_tiebreak"]
    assert not objective(cfg, root_z, dec_z, rev_z, decision_values=[0.0, 0.4, 0.7, 0.2],
                         decision_calls=calls)["dec_tiebreak"]


def test_advisor_failure_mid_grpo_is_fail_stop_and_resumes_identically(ckpt, tmp_path, monkeypatch):
    from mcq_rsi_helpers import advisor_server
    from src.manager.mcq_rsi.advisors import AdvisorError
    from src.verifiable.rsi_grpo import committed_step_directories
    anchors, rows = anchor_rows(tmp_path), make_rows(8)
    clean = G.train_fa_grpo(config(ckpt, steps=4), ckpt / "sft", rows, anchors, make_pool(tmp_path, "c1"),
                            tmp_path / "clean")
    broken = {"on": False}
    http, adapters = advisor_server(tmp_path, fail_when=lambda body: broken["on"] and body["model"] == "medqa_verifier")
    pool = CachedAdvisorPool("medqa", tmp_path / "c2", "http://x", adapters=adapters, http=http, sleep=lambda s: None)
    commit = G._commit_step

    def flip(root, step, *a, **k):
        name = commit(root, step, *a, **k)
        broken["on"] = step == 2
        return name

    monkeypatch.setattr(G, "_commit_step", flip)
    with pytest.raises(AdvisorError):
        G.train_fa_grpo(config(ckpt, steps=4), ckpt / "sft", rows, anchors, pool, tmp_path / "run")
    assert pool._abort.is_set() and len(committed_step_directories(tmp_path / "run")) == 2
    assert not (tmp_path / "run" / "summary.json").exists()
    monkeypatch.setattr(G, "_commit_step", commit)
    broken["on"] = False
    s = G.train_fa_grpo(config(ckpt, steps=4), ckpt / "sft", rows, anchors, pool, tmp_path / "run")  # same pool
    assert not pool._abort.is_set()
    assert s["final_adapter_sha256"] == clean["final_adapter_sha256"]
    assert [r["loss"] for r in s["per_step"]] == [r["loss"] for r in clean["per_step"]]


def test_guard_call_rate_delta_is_two_sided(monkeypatch):
    """|delta call rate| > 0.10 fails in both directions (a drop as much as a rise); KL_dec is checked too."""
    cfg = G.FAGRPOConfig()
    baseline = {"call_rate": 0.5}
    for rate, kl, passed in ((0.25, 0.0, False), (0.39, 0.0, False), (0.75, 0.0, False), (0.61, 0.0, False),
                             (0.45, 0.0, True), (0.55, 0.0, True), (0.5, 0.06, False), (0.5, 0.04, True)):
        monkeypatch.setattr(G, "guard_probe", lambda *a, r=rate, k=kl, **kw: {"n": 64, "call_rate": r, "actions": {},
                                                                               "kl_dec": k, "kl_root": 0.0})
        g = G._guard(None, None, [], baseline, cfg, 8)
        assert g["passed"] is passed and g["call_rate_delta"] == pytest.approx(rate - 0.5), (rate, kl)
        assert g["enforced"] and g["step"] == 8
        assert bool([r for r in g["reasons"] if r.startswith("|delta call rate|")]) is (abs(rate - 0.5) > 0.10)


def test_guard_call_rate_drop_rolls_back(ckpt, tmp_path, monkeypatch):
    """A real policy probe whose call rate is > 0.10 below S_k's, at the default threshold, rolls back to S_k.

    The tiny S_k commits on every probe root (call rate 0, so nothing can drop), so S_k's recorded
    baseline is raised to 0.5; the policy's probe, ``_guard`` and the rollback are the real ones.
    """
    real = G.guard_probe

    def probe(model, scorer, rows, adapter=G.POLICY, reference=G.REFERENCE):
        out = real(model, scorer, rows, adapter, reference)
        return {**out, "call_rate": 0.5} if reference is None else out  # the S_k baseline only

    monkeypatch.setattr(G, "guard_probe", probe)
    s = run(ckpt, tmp_path, steps=4)
    guard = s["guards"][0]
    assert s["baseline"]["call_rate"] == 0.5 and guard["call_rate"] == 0.0 and guard["call_rate_delta"] == -0.5
    assert s["config"]["guard_max_call_rate_delta"] == 0.10
    assert s["steps"] == 2 and s["guard_failed_step"] == 2 and s["selected_step"] == 0 == s["rollback_step"]
    assert s["final_adapter_sha256"] == sha(ckpt / "sft" / "adapter_model.safetensors")
    assert [r for r in guard["reasons"] if r.startswith("|delta call rate|")]
    assert not [r for r in guard["reasons"] if r.startswith("KL_dec")]


def test_question_objective_decomposition():
    """loss = beta_root KL_root + beta_dec KL_dec + mean_a PG_a + beta_rev mean_a KL_rev,a (distinct betas, large KLs)."""
    import torch
    root_z, dec_z, rev_z = leaves()
    cfg = G.FAGRPOConfig(beta_root=0.3, beta_dec=0.7, beta_rev=0.11)
    parts = objective(cfg, root_z, dec_z, rev_z, gold=2)
    lsm = lambda z: torch.log_softmax(z.detach(), 0)
    ref = lambda z: torch.log_softmax(z.detach() + 0.3 * torch.arange(len(z), dtype=z.dtype), 0)
    kl = lambda z: float(G.exact_kl(lsm(z), ref(z)))
    pgs = []
    for z in rev_z:
        p = lsm(z).exp()
        J = float(p[2])
        A = [((i == 2) - J) / (math.sqrt(J * (1 - J)) + 1e-4) for i in range(len(p))]
        pgs.append(-sum(float(p[i]) * A[i] * float(lsm(z)[i]) for i in range(len(p))))
    assert parts["J"] == pytest.approx([float(lsm(z).exp()[2]) for z in rev_z], abs=1e-12)
    assert float(parts["kl_root"]) == pytest.approx(kl(root_z), abs=1e-12) and kl(root_z) > 1e-3
    assert float(parts["kl_dec"]) == pytest.approx(kl(dec_z), abs=1e-12) and kl(dec_z) > 1e-3
    kl_rev = sum(kl(z) for z in rev_z) / 3
    assert float(parts["kl_rev"]) == pytest.approx(kl_rev, abs=1e-12) and kl_rev > 1e-3
    assert float(parts["pg"]) == pytest.approx(sum(pgs) / 3, abs=1e-12)
    expected = 0.3 * kl(root_z) + 0.7 * kl(dec_z) + sum(pgs) / 3 + 0.11 * kl_rev
    assert float(parts["loss"]) == pytest.approx(expected, abs=1e-12)


def _fresh_policy(ckpt, **kw):
    import torch
    cfg = G.FAGRPOConfig.from_dict(config(ckpt, **kw))
    cfg.validate()
    tok, model = G.load_policy(cfg, ckpt / "sft")
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    return cfg, tok, model, params, torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def test_train_step_wiring_matches_brute_force(ckpt, tmp_path, monkeypatch):
    """One real ``_train_step`` against brute force under S_k: the greedy root X, J = pi(gold | post-tool state)
    with the advisor's cached output, the PG value, the logged loss decomposition, lambda * grad(anchor CE)
    and gradient clipping."""
    import torch
    from peft import PeftModel
    from src.subagents.train import load_text_causal_model
    kinds = ["extractor", "reasoner", "verifier"]
    rows = make_rows(8)
    batch1, batch2 = rows[1:3], rows[5:7]
    assert all(r["ground_truth"] != "A" for r in batch1 + batch2)  # gold index != 0
    anchors = anchor_rows(tmp_path)
    pool = make_pool(tmp_path)
    clip = torch.nn.utils.clip_grad_norm_
    seen = []

    def spy(params, max_norm, *a, **k):
        params = list(params)
        grads = [p.grad.detach().clone() for p in params]
        pre = float(torch.sqrt(sum((g.double() ** 2).sum() for g in grads)))
        out = clip(params, max_norm, *a, **k)
        post = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params)))
        seen.append({"grads": grads, "pre": pre, "post": post, "max_norm": max_norm})
        return out

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)

    def step_one(with_anchor):
        cfg, tok, model, params, opt = _fresh_policy(ckpt, max_grad_norm=1e-3)
        scorer = G._Scorer(tok, MEDQA, kinds)
        features, _ = G.anchor_features(anchors, tok, cfg)
        batch = features[:cfg.anchor_rows_per_step] if with_anchor else []
        report = G._train_step(1, model, tok, scorer, opt, params, batch1, batch, pool, cfg)
        return cfg, tok, model, params, opt, scorer, features, report

    cfg, tok, model, params, opt, scorer, features, r1 = step_one(True)
    anchor_batch = features[:cfg.anchor_rows_per_step]
    grads_with = seen[-1]["grads"]

    # Brute force under S_k (an independent fp32 PeftModel): full-vocabulary sequence log-probs of each key path.
    sk = PeftModel.from_pretrained(load_text_causal_model(str(ckpt / "base"), dtype=torch.float32,
                                                          trust_remote_code=True), str(ckpt / "sft")).eval()

    def key_probs(messages, keys):
        ids = list(tok(P.render(tok, messages, P.TOOLS_DEPLOY))["input_ids"])
        paths = P.draft_paths(tok, keys)
        logp = []
        with torch.no_grad():
            for k in keys:
                seq = ids + list(paths[k])
                lp = torch.log_softmax(sk(input_ids=torch.tensor([seq])).logits[0].double(), -1)
                logp.append(sum(float(lp[len(ids) - 1 + t, token]) for t, token in enumerate(paths[k])))
        return torch.softmax(torch.tensor(logp, dtype=torch.float64), 0).tolist()

    assert [q["example_id"] for q in r1["questions"]] == [r["example_id"] for r in batch1]
    for row, q in zip(batch1, r1["questions"]):
        keys, eid = list(row["choices"]), int(row["example_id"])
        gold = keys.index(row["ground_truth"])
        root = key_probs(P.manager_messages(MEDQA, row), keys)
        assert q["draft"] == keys[max(range(len(keys)), key=lambda i: root[i])]  # greedy root X under S_k
        pgs = []
        for kind in kinds:
            output = pool.call(agent_kind=kind, example_id=eid, question=row["question"], context=row.get("context") or "",
                               choices=row["choices"], cache_namespace="mcq_rsi_grpo",
                               candidate_answer=q["draft"] if kind == "verifier" else "")
            assert output and "error" not in output
            state = P.manager_messages(MEDQA, row) + [P.eval_call_message(kind, q["draft"], eid, f"eval_{eid}_0"),
                                                      P.tool_message(kind, f"eval_{eid}_0", output)]
            pi = key_probs(state, keys)
            assert pi != pytest.approx(root, abs=1e-6)  # the post-tool state is not the root state
            assert q["J"][kind] == pytest.approx(pi[gold], abs=2e-5), kind
            J = pi[gold]
            pgs.append(-sum(pi[y] * ((y == gold) - J) / (math.sqrt(J * (1 - J)) + cfg.advantage_eps) * math.log(pi[y])
                            for y in range(len(keys))))
        assert q["pg"] == pytest.approx(sum(pgs) / len(kinds), abs=2e-5)
        assert q["kl_root"] == 0.0 and q["kl_dec"] == 0.0 and q["kl_rev"] == 0.0  # step 1: policy == S_k
        assert q["loss"] == pytest.approx(q["pg"], abs=1e-7)

    # Anchor: route-only CE (token mean) under S_k, and the logged step loss.
    ce, total = 0.0, 0
    for f in anchor_batch:
        with torch.no_grad():
            lp = torch.log_softmax(sk(input_ids=torch.tensor([f["input_ids"]])).logits[0].double(), -1)
        for i, y in enumerate(f["labels"][1:]):
            if y != -100:
                ce -= float(lp[i, y])
                total += 1
    assert r1["anchor_supervised_tokens"] == total and r1["anchor_ce"] == pytest.approx(ce / total, rel=1e-5)
    mean_q = sum(q["loss"] for q in r1["questions"]) / 2
    assert r1["loss"] == pytest.approx(mean_q + cfg.anchor_lambda * r1["anchor_ce"], abs=1e-7)

    # Clipping: the logged norm is the pre-clip norm; the update used gradients clipped to max_grad_norm.
    first = seen[0]
    assert first["max_norm"] == cfg.max_grad_norm == 1e-3 and first["pre"] > 10 * cfg.max_grad_norm
    assert r1["grad_norm"] == pytest.approx(first["pre"], rel=1e-5)
    assert first["post"] == pytest.approx(cfg.max_grad_norm, rel=1e-4)

    # The anchor's share of the gradient is exactly anchor_lambda * grad(CE token mean) under S_k.
    step_one(False)
    diff = [a - b for a, b in zip(grads_with, seen[-1]["grads"])]
    _, _, m2, p2, _ = _fresh_policy(ckpt)
    loss = 0
    for f in anchor_batch:
        logits = m2(input_ids=torch.tensor([f["input_ids"]])).logits[0, :-1].float()
        loss = loss + torch.nn.functional.cross_entropy(logits, torch.tensor(f["labels"][1:]), ignore_index=-100,
                                                        reduction="sum")
    expected = torch.autograd.grad(cfg.anchor_lambda * loss / total, p2)
    norm = lambda gs: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs)))
    assert norm(expected) > 0
    assert norm([d - e for d, e in zip(diff, expected)]) <= 1e-3 * norm(expected)

    # Step 2 (policy != S_k): the logged loss decomposes with every beta and the KLs are positive.
    r2 = G._train_step(2, model, tok, scorer, opt, params, batch2, features[cfg.anchor_rows_per_step:2 * cfg.anchor_rows_per_step],
                       pool, cfg)
    tol = 1e-7
    beta_terms = 0.0
    for q in r2["questions"]:
        assert q["kl_root"] > 0 and q["kl_dec"] > 0 and q["kl_rev"] > 0
        kls = cfg.beta_root * q["kl_root"] + cfg.beta_dec * q["kl_dec"] + cfg.beta_rev * q["kl_rev"]
        assert q["loss"] == pytest.approx(q["pg"] + kls, abs=tol)
        beta_terms += min(cfg.beta_root * q["kl_root"] + cfg.beta_dec * q["kl_dec"], cfg.beta_rev * q["kl_rev"])
    assert beta_terms > 20 * tol  # each beta term is far above the tolerance: dropping one is detected
    assert r2["loss"] == pytest.approx(sum(q["loss"] for q in r2["questions"]) / 2
                                       + cfg.anchor_lambda * r2["anchor_ce"], abs=1e-7)
    assert seen[-1]["post"] == pytest.approx(min(seen[-1]["pre"], cfg.max_grad_norm), rel=1e-4)


def test_prune_resume_and_rollback_together(ckpt, tmp_path, monkeypatch):
    """Pruning keeps exactly what the resume pointer (newest step: adapter + optimizer) and the rollback
    (last passing guarded step: adapter) need, across a crash, a resume and a guard failure."""
    real_guard = G._guard

    def scripted(model, scorer, probe, baseline, cfg, step):
        out = real_guard(model, scorer, probe, baseline, cfg, step)
        return {**out, "passed": step != 4, "reasons": [] if step != 4 else ["scripted failure"]}

    monkeypatch.setattr(G, "_guard", scripted)
    clean = run(ckpt, tmp_path, out="clean", steps=6)
    assert clean["guard_failed_step"] == 4 and clean["selected_step"] == 2
    step = lambda out, i: next((tmp_path / out).glob(f"step-{i:05d}-*"))
    files = lambda d: set(p.name for p in d.iterdir())
    clean_step2, clean_step4 = (sha(step("clean", i) / "adapter_model.safetensors") for i in (2, 4))
    assert clean["final_adapter_sha256"] == clean_step2

    advance = G._advance_pointer

    def crash_at_4(root, names, i):
        if i == 4:
            raise Crash("killed after step 4 was renamed, before the pointer")
        return advance(root, names, i)

    monkeypatch.setattr(G, "_advance_pointer", crash_at_4)
    with pytest.raises(Crash):
        run(ckpt, tmp_path, out="crash", steps=6)
    out = tmp_path / "crash"
    s1, s2, s3 = (step("crash", i) for i in (1, 2, 3))
    orphan = step("crash", 4)
    assert json.loads((out / "resume.json").read_text())["directory"] == s3.name
    # After step 3: step 1 holds no weights; step 2 (guard passed at 2: the rollback target) keeps its adapter
    # but no optimizer; step 3 (the resume pointer) keeps both.
    assert not files(s1) & {"adapter_model.safetensors", "optimizer.pt"} and "step.json" in files(s1)
    assert "adapter_model.safetensors" in files(s2) and "optimizer.pt" not in files(s2)
    assert {"adapter_model.safetensors", "optimizer.pt"} <= files(s3)
    assert sha(s2 / "adapter_model.safetensors") == clean_step2

    monkeypatch.setattr(G, "_advance_pointer", advance)
    s = run(ckpt, tmp_path, out="crash", steps=6)  # resume from step 3's adapter + optimizer, redo step 4
    assert not orphan.exists() and s["steps"] == 4 and s["guard_failed_step"] == 4 and s["selected_step"] == 2
    assert [r["loss"] for r in s["per_step"]] == [r["loss"] for r in clean["per_step"]]
    assert s["final_adapter_sha256"] == clean_step2 == sha(s2 / "adapter_model.safetensors")
    s4 = step("crash", 4)
    assert sha(s4 / "adapter_model.safetensors") == clean_step4  # the resumed optimizer state was step 3's
    # Final state: the newest step and the rollback target keep adapters; no optimizer state anywhere.
    assert "adapter_model.safetensors" not in files(s3) and "adapter_model.safetensors" in files(s4)
    assert not list(out.rglob("optimizer.pt")) and not list(out.glob("incomplete-*"))
    assert all("step.json" in files(d) for d in (s1, s2, s3, s4))
    assert s == run(ckpt, tmp_path, out="crash", steps=6)


def test_step_commit_is_durable_before_the_pointer(ckpt, tmp_path, monkeypatch):
    """Every file and directory of a step is fsynced before its rename; the parent directory is fsynced
    after the rename and before the resume pointer is written, and again after the pointer's rename."""
    import os
    events = []
    real_tree, real_dir, real_replace, real_fsync = G._fsync_tree, G._fsync_dir, os.replace, os.fsync
    monkeypatch.setattr(G, "_fsync_tree", lambda p: (events.append(("tree", Path(p).name)), real_tree(p))[1])
    monkeypatch.setattr(G, "_fsync_dir", lambda p: (events.append(("dir", Path(p).name)), real_dir(p))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (events.append(("replace", Path(a).name, Path(b).name)),
                                                     real_replace(a, b))[1])
    run(ckpt, tmp_path, steps=2)
    for i in (1, 2):
        rename = next(k for k, e in enumerate(events) if e[0] == "replace" and e[2].startswith(f"step-{i:05d}-"))
        stage = events[rename][1]
        assert stage.startswith("incomplete-")
        tree = events.index(("tree", stage))
        assert tree < rename and ("dir", stage) in events[tree:rename]
        pointer = next(k for k, e in enumerate(events) if k > rename and e[0] == "replace" and e[2] == "resume.json")
        assert ("dir", "run") in events[rename + 1:pointer]
        assert events[pointer + 1] == ("dir", "run")
    # _fsync_tree fsyncs every file and every directory under the step.
    tree = tmp_path / "t"
    (tree / "a").mkdir(parents=True)
    (tree / "x").write_text("1")
    (tree / "a" / "y").write_text("2")
    calls = []
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real_fsync(fd))[1])
    real_tree(tree)
    assert len(calls) == 4  # two files, two directories
