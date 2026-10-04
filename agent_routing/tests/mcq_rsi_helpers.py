"""Shared offline fakes for the MCQ RSI collector tests (no downloads, CPU only)."""
import functools
import hashlib
import json
import os
import re
from pathlib import Path

from src.manager.marginal_value import ADVISOR_KINDS, _draft_and_final, _draft_only
from src.manager.mcq_rsi.benchmarks import BASE_MODEL
from src.manager.mcq_rsi.protocol import Decision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mcq_rsi"
ROOT = Path(__file__).resolve().parents[1]


def real_tokenizer_dir():
    """Qwen3.5 tokenizer + stock template: MCQ_RSI_TOKENIZER_DIR or the imported S_1 adapter folder."""
    default = Path(os.environ.get("MCQ_RSI_IMPORT_DIR", ROOT / "outputs/mcq_rsi/import")) / "medqa/round1/sft"
    path = Path(os.environ.get("MCQ_RSI_TOKENIZER_DIR", default))
    return path if (path / "tokenizer.json").exists() and (path / "chat_template.jinja").exists() else None


@functools.lru_cache(maxsize=1)
def tiny_tokenizer():
    """Byte-level BPE (exact round trip) with Qwen's special tokens and the stock Qwen3.5 chat template."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<think>", "</think>",
               "<tool_response>", "</tool_response>"]
    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.decoder = decoders.ByteLevel()
    corpus = ["DRAFT_ANSWER_A\nANSWER_A DRAFT_ANSWER_B\nANSWER_B", "<function=extractor_tool>\n</function>",
              "<parameter=current_draft>\nB\n</parameter>", "verifier_tool reasoner_tool extractor_tool",
              "Question Choices Context Example ID system user assistant tool manager"] * 20
    raw.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=special, show_progress=False,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=special[1:])
    tok.chat_template = (FIXTURES / "qwen3_5_chat_template.jinja").read_text(encoding="utf-8")
    return tok


def tiny_model(tok, seed=0):
    """Randomly initialised two-layer Qwen3.5 (one linear-attention, one full-attention layer)."""
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                            layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2,
                            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return Qwen3_5ForCausalLM(cfg).eval()


def make_rows(n=4, keys="ABCD", start=100):
    return [{"example_id": start + i, "benchmark_name": "medqa", "task_subtype": "t",
             "question": f"Question number {start + i} about a patient?", "context": "",
             "choices": {k: f"option {k.lower()} {i}" for k in keys}, "ground_truth": keys[i % len(keys)],
             "split": "train"} for i in range(n)]


def sequence_of(messages):
    """Advisor kinds called so far in a branch state (marginal_value call turns)."""
    return tuple(c["function"]["name"][:-5] for m in messages if m.get("role") == "assistant"
                 for c in m.get("tool_calls") or [])


def example_id_of(messages):
    return int(re.search(r"Example ID: (\d+)", messages[1]["content"]).group(1))


class FakeManager:
    """Scripted manager. ``script[eid]`` = {"root": (key, action), "rev": {sequence tuple: key or (draft, answer)}}.

    Unscripted revisions keep the current draft. ``fail_at`` raises on that root (a simulated kill).
    """

    def __init__(self, script, fail_at=None, name="fake"):
        self.script, self.fail_at, self.name = script, fail_at, name
        self.calls = {"root": [], "revise": [], "probe": []}

    def identity(self):
        return {"fake": self.name}

    def _rev(self, messages):
        eid, seq = example_id_of(messages), sequence_of(messages)
        draft = [m for m in messages if m.get("role") == "assistant"][-1]["content"].split("_")[-1]
        return self.script.get(eid, {}).get("rev", {}).get(seq, draft)

    def root(self, messages, keys, kinds, example_id):
        self.calls["root"].append(example_id)
        if example_id == self.fail_at:
            raise KeyboardInterrupt("simulated kill")
        key, action = self.script[example_id]["root"]
        kind = None if action == "commit" else action
        text = _draft_and_final(key) if kind is None else _draft_only(key) + "\n\n<tool_call>..."
        return Decision(key, kind, text, -0.25, {"checked": 6, "mismatch": 0, "first_mismatch": None},
                        {k: -1.0 for k in keys})

    def revise(self, states, keys):
        self.calls["revise"].append(len(states))
        self.calls.setdefault("revise_states", []).extend(states)
        out = []
        for m in states:
            y, z = (lambda r: r if isinstance(r, tuple) else (r, r))(self._rev(m))
            out.append(Decision(z, None, f"DRAFT_ANSWER_{y}\nANSWER_{z}", -0.5,
                                {"checked": 9, "mismatch": 0, "first_mismatch": None, "would_call": int(y != z)},
                                draft=y))
        return out

    def probe(self, messages, keys, probe):
        self.calls["probe"].append(sequence_of(messages))
        if not sequence_of(messages):
            key = self.script[example_id_of(messages)]["root"][0]
        else:
            key = self._rev(messages)
            key = key[1] if isinstance(key, tuple) else key
        return key, _draft_and_final(key), True


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._body


class FakeVLLM:
    """OpenAI-compatible fake: deterministic advisor text from the request; scripted failures.

    ``text(body)`` overrides the reply text; ``hold(body)`` returns an Event the request waits on.
    """

    def __init__(self, served=None, fail=0, fail_when=None, text=None, hold=None, version=None):
        self.served = served
        self.fail, self.fail_when, self.text, self.hold = fail, fail_when, text, hold
        self.version = version  # ``GET /version`` body (None: answered like /v1/models)
        self.posts, self.gets = [], []
        import threading
        self.lock = threading.Lock()

    def get(self, url, timeout=None):
        self.gets.append(url)
        if url.endswith("/version") and self.version is not None:
            return FakeResponse(200, self.version)
        return FakeResponse(200, {"data": [m if isinstance(m, dict) else {"id": m} for m in (self.served or [])]})

    def post(self, url, json=None, timeout=None):
        with self.lock:
            self.posts.append(json)
            failing = self.fail > 0 or (self.fail_when and self.fail_when(json))
            if self.fail > 0:
                self.fail -= 1
        if self.hold and self.hold(json) is not None:
            self.hold(json).wait(10)
        if failing:
            return FakeResponse(500, {})
        user = json["messages"][-1]["content"]
        text = self.text(json) if self.text else f" {json['model']}|{len(user)}|{user[-12:]} "
        return FakeResponse(200, {"model": json["model"], "choices": [{"message": {"content": text},
                                                                        "finish_reason": "stop"}]})


def served(bench):
    return [f"{bench}_{k}" for k in ADVISOR_KINDS] + ["base"]


def fake_adapters(tmp_path, bench="medqa"):
    """Fake adapter directories: (vLLM ``/v1/models`` LoRA cards, pinned identities for ``CachedAdvisorPool``)."""
    cards, adapters = [], {}
    for kind in ADVISOR_KINDS:
        root = Path(tmp_path) / "adapters" / bench / kind
        root.mkdir(parents=True, exist_ok=True)
        (root / "adapter_model.safetensors").write_bytes(f"{bench} {kind} weights".encode())
        sha = hashlib.sha256((root / "adapter_model.safetensors").read_bytes()).hexdigest()
        cards.append({"id": f"{bench}_{kind}", "root": str(root), "parent": BASE_MODEL})
        adapters[kind] = f"test://{bench}/{kind}#sha256={sha}"
    return cards + [{"id": BASE_MODEL, "root": BASE_MODEL, "parent": None}], adapters


def advisor_server(tmp_path, bench="medqa", **kw):
    """A FakeVLLM serving the fake adapters, plus the matching ``adapters`` argument."""
    cards, adapters = fake_adapters(tmp_path, bench)
    return FakeVLLM(cards, **kw), adapters


def read_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))
