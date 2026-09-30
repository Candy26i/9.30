import string

import pytest

from src.verifiable.actions import ActionTrie, decision_paths
from src.verifiable.protocol import COMMIT, call_message, parse_calls, tool_schemas
from src.verifiable.training import tokenize_turn


def character_tokenizer():
    from tokenizers import Tokenizer, pre_tokenizers, decoders
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from src.verifiable.backend import configure_tokenizer
    words = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", *string.printable]
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    raw.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>",
        eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    return configure_tokenizer(tok)


def test_grammar_requires_closure_and_excludes_repeated_advisor():
    tok = character_tokenizer()
    history = [{"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "reasoner_tool", "arguments": {}}}]}]
    paths = decision_paths(tok, history, tool_schemas(), 128)
    texts = [tok.decode(p[:-1]) for p in paths]
    # COMMIT plus two surface forms for each of the two unused tools.
    assert len(texts) == 5 and "COMMIT" in texts
    assert not any("reasoner_tool" in text for text in texts)
    for text in texts:
        parse_calls(text)
    trie = ActionTrie(paths)
    path = next(p for p in paths if tok.decode(p[:-1]).lstrip("\n").startswith("<tool_call>"))
    assert tok.eos_token_id not in trie.allowed(path[:10])
    assert trie.allowed(path[:-1]) == [tok.eos_token_id]
    with pytest.raises(ValueError, match="budget"):
        decision_paths(tok, [{"role": "user", "content": "Q"}], tool_schemas(), 2)


def _supervised_decision_ids(tok, history, response):
    decision_type = "call" if response.get("tool_calls") else "commit"
    encoded = tokenize_turn({"prompt": history, "response": [response], "decision_type": decision_type}, tok, 4096)
    target = [t for t, label in zip(encoded["input_ids"], encoded["labels"]) if label != -100]
    newline = tok("\n", add_special_tokens=False)["input_ids"]
    # The ChatML newline after EOS is supervised but never generated.
    assert target[-len(newline):] == newline
    return target[:-len(newline)]


def test_grammar_admits_supervised_sft_targets_and_instructed_compact_calls():
    tok = character_tokenizer()
    depth1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
              {"role": "assistant", "content": "CANDIDATE_ANSWER: 1"}, {"role": "user", "content": "Decide."}]
    depth2 = depth1 + [call_message("reasoner", "draft", "call_reasoner"),
                       {"role": "tool", "tool_call_id": "call_reasoner", "name": "reasoner_tool", "content": "Advice."},
                       {"role": "user", "content": "Revise."},
                       {"role": "assistant", "content": "CANDIDATE_ANSWER: 2"}, {"role": "user", "content": "Decide."}]
    for history, kinds in ((depth1, ("extractor", "reasoner", "verifier")), (depth2, ("extractor", "verifier"))):
        paths = decision_paths(tok, history, tool_schemas(), 128)
        assert len(paths) == 1 + 2 * len(kinds)
        responses = [{"role": "assistant", "content": COMMIT}] + [
            call_message(kind, "draft", "call_" + kind) for kind in kinds]
        for response in responses:
            assert _supervised_decision_ids(tok, history, response) in paths
        for kind in kinds:
            compact = '<tool_call>{"name":"%s_tool","arguments":{}}</tool_call>' % kind
            assert tok(compact, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] in paths
        decoded = [tok.decode(path[:-1]) for path in paths]
        assert all(parse_calls(text)[1] or text == COMMIT for text in decoded)
        assert {call["name"] for text in decoded for call in parse_calls(text)[1]} == {k + "_tool" for k in kinds}


def test_real_constrained_sampling_and_scoring_are_identical():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from src.verifiable.rsi_grpo import RolloutBackend, turn_logprobs
    torch.set_num_threads(1)
    tok = character_tokenizer()
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=4096, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id))
    model.eval()
    backend = RolloutBackend(tok, model, {"max_context": 4096, "max_seq_len": 4096,
        "rl_temperature": .8, "decision_constraint": "finite_actions_v1"})
    backend.turns = []
    original = model.generate
    captured = {}
    def record(**kwargs):
        out = original(**kwargs, return_dict_in_generate=True, output_scores=True)
        captured["scores"] = out.scores
        return out.sequences
    model.generate = record
    result = backend.generate([{"role": "user", "content": "Choose an action."}], tools=tool_schemas(), max_tokens=128)
    assert not result["truncated"]
    content, calls = parse_calls(result["text"])
    assert content == "COMMIT" or (not content and len(calls) == 1)
    turn = backend.turns[0]
    scored = turn_logprobs(model, turn, .8)
    sampled = torch.stack([torch.log_softmax(scores[0].float(), -1)[token]
                           for scores, token in zip(captured["scores"], turn["completion_ids"])])
    assert torch.allclose(scored, sampled, atol=1e-5), (scored, sampled)
    (-scored.sum()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # The same grammar also applies to greedy evaluation via HFBackend.
    backend.turns = None
    greedy = backend.generate([{"role": "user", "content": "Choose."}], tools=tool_schemas(), max_tokens=128)
    content, calls = parse_calls(greedy["text"])
    assert content == "COMMIT" or (not content and len(calls) == 1)
