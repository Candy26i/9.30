import math
import re

import pytest

from mcq_rsi_helpers import real_tokenizer_dir, tiny_model, tiny_tokenizer
from src.manager import marginal_value
from src.manager.marginal_value import _draft_and_final, _draft_only
from src.manager.mcq_rsi import benchmarks as registry
from src.manager.mcq_rsi import protocol as P
from src.pipeline import stages

MSGS = [{"role": "system", "content": "You are a manager agent."}, {"role": "user", "content": "Example ID: 7\n\nQuestion one?"}]
KEYS = list("ABCD")


def test_tool_schemas_are_the_paper_ones():
    assert P.TOOLS_DEPLOY == stages._manager_tool_schemas("environment")
    assert P.TOOLS_SFT == marginal_value._tool_schemas("environment")
    names = lambda tools: [(t["function"]["name"], t["function"]["parameters"]) for t in tools]
    assert names(P.TOOLS_DEPLOY) != names(P.TOOLS_SFT)  # descriptions of current_draft differ
    assert [n for n, _ in names(P.TOOLS_DEPLOY)] == [n for n, _ in names(P.TOOLS_SFT)]


def test_deploy_and_sft_renders_differ_only_in_tool_block():
    tok = tiny_tokenizer()
    block = re.compile(r"<tools>.*</tools>", re.S)
    deploy, sft = P.render(tok, MSGS, P.TOOLS_DEPLOY), P.render(tok, MSGS, P.TOOLS_SFT)
    assert deploy != sft and block.sub("", deploy) == block.sub("", sft)
    assert deploy.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Extract decision-relevant factual signals from the question and context." in deploy
    assert P.render(tok, MSGS, P.TOOLS_DEPLOY) == stages._render_manager_chat(tok, MSGS, P.TOOLS_DEPLOY)


def test_manager_messages_rebuild_paper_base_messages():
    from mcq_rsi_helpers import read_fixture
    fx = read_fixture("paper_medqa_replay.json")
    bench = registry.get("medqa")
    for row, record in zip(fx["rows"], fx["records"]):
        assert P.manager_messages(bench, row) == record["base_messages"]


def test_actions_trie_and_eval_parsing():
    tok = tiny_tokenizer()
    actions = P.decision_actions(tok, MSGS, P.TOOLS_DEPLOY, KEYS, example_id=7)
    assert len(actions) == 16 and len({a.ids for a in actions}) == 16
    trie = P.make_trie(a.ids for a in actions)
    assert all(a.ids[-1] == tok.eos_token_id for a in actions)
    commit = next(a for a in actions if a.key == "B" and a.kind is None)
    verifier = next(a for a in actions if a.key == "B" and a.kind == "verifier")
    assert commit.text == _draft_and_final("B")
    assert verifier.text.startswith(_draft_only("B") + "\n\n<tool_call>\n<function=verifier_tool>")
    assert "<parameter=current_draft>\nB\n</parameter>" in verifier.text
    assert P.complete_path(trie, list(commit.ids) + [0]) == commit.ids
    # Recorded eval turns (environment binding injects example_id) map back onto the grammar.
    assert P.parse_decision("DRAFT_ANSWER_B\nANSWER_B", None, KEYS) == ("B", None)
    assert P.parse_decision("DRAFT_ANSWER_B", {"name": "verifier_tool", "arguments": {"example_id": 7, "current_draft": "B"}},
                            KEYS) == ("B", "verifier")
    assert P.parse_decision("DRAFT_ANSWER_B", {"name": "verifier_tool", "arguments": {"current_draft": "C"}}, KEYS) is None
    assert P.parse_decision("I think B\nANSWER_B", None, KEYS) is None
    with pytest.raises(ValueError):
        P.make_trie([(1, 2), (1, 2, 3)])


def test_key_token_analysis_tiny_tokenizer():
    a = P.key_token_analysis(tiny_tokenizer(), KEYS)
    assert a["single_token_keys"] and a["sft_target_prefix"]


@pytest.mark.skipif(real_tokenizer_dir() is None, reason="Qwen3.5 tokenizer not present (set MCQ_RSI_TOKENIZER_DIR)")
def test_real_qwen35_key_tokens():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(real_tokenizer_dir()))
    for tools in (P.TOOLS_DEPLOY, P.TOOLS_SFT):
        a = P.key_token_analysis(tok, list("ABCDEFGHIJ"), tools=tools)
        assert a["prefix"] == ["D", "RAFT", "_ANS", "WER"]
        assert a["key_tokens"] == {k: ["_" + k] for k in "ABCDEFGHIJ"} and a["single_token_keys"]
        assert a["diverge_after_key"] and a["commit_next"] == ["\n"] and a["call_next"] == ["\n\n"]
        assert a["sft_target_prefix"]
        assert sorted(set(a["action_lengths"].values())) == [10, 22, 34]  # commit, E/R call, V call (+EOS)
    # Forced revision DRAFT_ANSWER_Y\nANSWER_Z: Y and Z are each one token (both key distributions are exact).
    keys = list("ABCDEFGHIJ")
    paths = P.revision_paths(tok, keys)
    drafts = P.draft_paths(tok, keys)
    assert P.single_token_choice(drafts) is not None
    assert all(P.single_token_choice({z: paths[(y, z)] for z in keys}) is not None for y in keys)
    assert {len(p) for p in paths.values()} == {10}


@pytest.fixture(scope="module")
def tiny():
    import torch
    torch.set_num_threads(1)
    tok = tiny_tokenizer()
    return tok, tiny_model(tok)


def _next_logits(model, prompt):
    import torch

    def fn(prefix):
        with torch.no_grad():
            return model(input_ids=torch.tensor([list(prompt) + list(prefix)])).logits[0, -1]
    return fn


def test_trie_path_scoring_equals_brute_force(tiny):
    import torch
    tok, model = tiny
    backend = P.HFBackend(model, tok)
    actions = P.decision_actions(tok, MSGS, P.TOOLS_DEPLOY, KEYS[:2])
    trie = P.make_trie(a.ids for a in actions)
    prompt = backend.prompt_ids(MSGS, P.TOOLS_DEPLOY)
    with torch.no_grad():
        scores = P.score_paths(model, prompt, [a.ids for a in actions], trie, backend.pad_id)
    nxt = _next_logits(model, prompt)
    for action, score in zip(actions, scores.tolist()):
        brute = 0.0
        for t, token in enumerate(action.ids):
            allowed = trie.allowed(action.ids[:t])
            row = nxt(action.ids[:t]).double()
            brute += float(row[token] - torch.logsumexp(row[allowed], 0))
        assert score == pytest.approx(brute, abs=1e-4)
    assert float(torch.logsumexp(scores, 0)) == pytest.approx(0.0, abs=1e-4)  # a distribution over legal turns


def test_constrained_greedy_generation_matches_reference_and_batching(tiny):
    tok, model = tiny
    backend = P.HFBackend(model, tok, batch_size=3)
    manager = P.Manager(backend)
    actions = P.decision_actions(tok, MSGS, P.TOOLS_DEPLOY, KEYS)
    trie = P.make_trie(a.ids for a in actions)
    prompt = backend.prompt_ids(MSGS, P.TOOLS_DEPLOY)
    got = backend.greedy([prompt], [trie])[0]
    ref = P.constrained_greedy_walk(_next_logits(model, prompt), trie)
    assert got["path"] == ref["path"] and got["logprob"] == pytest.approx(ref["logprob"], abs=1e-4)
    assert got["check"] == ref["check"]
    decision = manager.root(MSGS, KEYS, ("extractor", "reasoner", "verifier"), 7)
    assert next(a for a in actions if a.ids == got["path"]).text == decision.text
    assert math.isclose(sum(math.exp(v) for v in decision.key_logprobs.values()), 1.0, abs_tol=1e-4)
    # Revisions batched with left padding equal one-by-one revisions.
    states = [MSGS, MSGS[:1] + [{"role": "user", "content": "Example ID: 8\n\nA much longer question text here?"}],
              MSGS[:1] + [{"role": "user", "content": "Q"}]]
    batched = manager.revise(states, KEYS)
    single = [manager.revise([s], KEYS)[0] for s in states]
    assert [d.key for d in batched] == [d.key for d in single]
    assert [d.logprob for d in batched] == pytest.approx([d.logprob for d in single], abs=1e-4)
    assert all(d.text == P.revision_text(d.draft, d.key) for d in batched)
    assert all(math.isclose(sum(math.exp(v) for v in d.answer_logprobs.values()), 1.0, abs_tol=1e-4) for d in batched)


def test_unconstrained_argmax_check_agrees_with_free_greedy(tiny):
    import torch
    tok, model = tiny
    backend = P.HFBackend(model, tok)
    trie = P.make_trie(P.draft_paths(tok, KEYS).values())
    prompt = backend.prompt_ids(MSGS, P.TOOLS_DEPLOY)
    got = backend.greedy([prompt], [trie])[0]
    with torch.no_grad():
        free = model.generate(input_ids=torch.tensor([prompt]), attention_mask=torch.ones(1, len(prompt), dtype=torch.long),
                              max_new_tokens=len(got["path"]), do_sample=False, pad_token_id=backend.pad_id,
                              eos_token_id=tok.eos_token_id)[0, len(prompt):].tolist()
    diverge = next((i for i, (a, b) in enumerate(zip(free, got["path"])) if a != b), None)
    assert got["check"]["first_mismatch"] == diverge
    assert (got["check"]["mismatch"] == 0) == (tuple(free[:len(got["path"])]) == got["path"])


def test_constrained_equals_unconstrained_when_policy_is_in_grammar():
    import torch
    tok = tiny_tokenizer()
    actions = P.decision_actions(tok, MSGS, P.TOOLS_DEPLOY, KEYS)
    trie = P.make_trie(a.ids for a in actions)
    target = next(a for a in actions if a.key == "C" and a.kind == "reasoner").ids

    def peaked(off_grammar):
        def fn(prefix):
            row = torch.zeros(len(tok))
            row[target[len(prefix)]] = 5.0
            if off_grammar and len(prefix) == 2:
                row[tok.pad_token_id] = 9.0  # the free policy would leave the grammar here
            return row
        return fn

    ok = P.constrained_greedy_walk(peaked(False), trie)
    assert ok["path"] == target and ok["check"] == {"checked": len(target), "mismatch": 0, "first_mismatch": None}
    off = P.constrained_greedy_walk(peaked(True), trie)
    assert off["path"] == target and off["check"]["mismatch"] == 1 and off["check"]["first_mismatch"] == 2
    # merge_checks sums checked/mismatch and the revisions' would_call (0 for decisions without a forced token).
    assert P.merge_checks([ok["check"], off["check"], None]) == {"checked": 2 * len(target), "mismatch": 1,
                                                                 "would_call": 0}


def test_forced_commit_is_reported_as_would_call_not_mismatch():
    path = (5, 6, 7, 8)
    # Position 2 is the forced "\nANSWER" token after a revision: an intervention, not a policy mismatch.
    agree = P.path_check([5, 6, 7, 8], path, forced=2)
    leave = P.path_check([5, 6, 99, 8], path, forced=2)
    both = P.path_check([5, 1, 99, 8], path, forced=2)
    assert agree == {"checked": 3, "mismatch": 0, "first_mismatch": None, "would_call": 0}
    assert leave == {"checked": 3, "mismatch": 0, "first_mismatch": None, "would_call": 1}
    assert both == {"checked": 3, "mismatch": 1, "first_mismatch": 1, "would_call": 1}
    assert P.path_check([5, 6, 99, 8], path) == {"checked": 4, "mismatch": 1, "first_mismatch": 2}
    assert P.merge_checks([agree, leave, both]) == {"checked": 9, "mismatch": 1, "would_call": 2}


def test_revision_outcome_is_the_committed_answer_key():
    """Paper (``pred = final or draft``) and deployed eval (``parse_final_answer``) both score ANSWER_Z."""
    from src.manager.prompt import parse_final_answer
    text = P.revision_text("A", "C")
    assert text == "DRAFT_ANSWER_A\nANSWER_C"
    assert parse_final_answer(text, KEYS) == "C"
    tok = tiny_tokenizer()
    paths = P.revision_paths(tok, KEYS)
    assert len(paths) == 16 and all(tok.decode(ids[:-1]) == P.revision_text(y, z) for (y, z), ids in paths.items())


def test_revise_scores_the_committed_key_and_exempts_only_the_forced_token():
    """Peaked logits on DRAFT_ANSWER_A\\nANSWER_C: the outcome is Z=C (not the draft Y=A), and the free
    policy leaving the grammar at the forced commit token is a would_call, not an argmax mismatch."""
    import torch
    tok = tiny_tokenizer()
    paths, drafts = P.revision_paths(tok, KEYS), P.draft_paths(tok, KEYS)
    target = paths[("A", "C")]
    forced = len(drafts["A"])
    assert forced < len(target) - 1 and target[forced] != tok.pad_token_id

    def peaked(prefix):
        row = torch.zeros(len(tok))
        row[target[len(prefix)]] = 5.0
        if len(prefix) == forced:
            row[tok.pad_token_id] = 9.0  # the policy would not commit here (it would call)
        return row

    class PeakedBackend:
        tokenizer = tok

        def prompt_ids(self, messages, tools):
            return [0]

        def greedy(self, prompts, tries):
            return [P.constrained_greedy_walk(peaked, trie) for trie in tries]

    (d,) = P.Manager(PeakedBackend()).revise([MSGS], KEYS)
    assert (d.key, d.draft, d.kind, d.name) == ("C", "A", None, "commit")
    assert d.text == P.revision_text("A", "C") == "DRAFT_ANSWER_A\nANSWER_C"
    assert d.check == {"checked": len(target) - 1, "mismatch": 0, "first_mismatch": None, "would_call": 1}
    assert max(d.key_logprobs, key=d.key_logprobs.get) == "A"
    assert max(d.answer_logprobs, key=d.answer_logprobs.get) == "C"
