"""Manager decision interface for MCQ RSI under the stock Qwen3.5 template.

The paper-era manager answers every decision turn with one of finitely many
assistant texts, rendered by the template exactly as Manager SFT supervises them:

    commit(K):   DRAFT_ANSWER_K\\nANSWER_K
    call(a, K):  DRAFT_ANSWER_K + native tool call of a (Verifier: current_draft=K)

each followed by EOS. Every legal turn is a token path in an ``ActionTrie``;
greedy decoding restricted to the trie is the grammar-constrained policy, and a
path's probability is the product of per-token softmaxes renormalised over the
trie's allowed set (what constrained sampling draws; it sums to 1 over paths).

With the real Qwen3.5 tokenizer ``DRAFT_ANSWER_K`` is ``D RAFT _ANS WER _K`` for
every K in A-J: the key is one token after a shared 4-token prefix, and commit
(``\\n ANS WER _K``) and call (``\\n\\n <tool_call> ...``) diverge at the very next
token (``key_token_analysis``; tests/test_mcq_rsi_protocol.py).

A revision (after a forced call) forces the commit: ``DRAFT_ANSWER_Y\\nANSWER_Z``
with Y and Z each chosen greedily among the keys and ``\\nANSWER`` forced. Its
outcome is Z, as in the paper (``pred = final or draft``) and the deployed eval
(``parse_final_answer``); Y != Z is rare and reported.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...pipeline.stages import _manager_tool_schemas
from ...verifiable.actions import ActionTrie
from ..marginal_value import (
    ADVISOR_KINDS,
    _answer_token,
    _draft_and_final,
    _draft_only,
    _generate_answer,
    _normalize_tool_calls_mv,
    _render_chat,
    _tool_call_message,
    _tool_schemas,
)
from ..prompt import build_manager_user_message

PROTOCOL_VERSION = "mcq_rsi_protocol/1"
BINDING = "environment"
TOOLS_DEPLOY = _manager_tool_schemas(BINDING)  # eval / collection / GRPO (deployment state)
TOOLS_SFT = _tool_schemas(BINDING)  # paper Manager SFT and probe-mode collection


# ------------------------------------------------------------------ rendering

def manager_messages(bench, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """(system, user) of the paper manager, environment binding (records' ``base_messages``)."""
    keys = list(row["choices"])
    return [
        {"role": "system", "content": bench.manager_system_prompt(keys)},
        {"role": "user", "content": build_manager_user_message(
            example_id=int(row["example_id"]), question=row["question"], context=row.get("context") or "",
            choices=row["choices"], binding_mode=BINDING)},
    ]


def render(tokenizer, messages, tools) -> str:
    """Prompt with the generation prefix (``<think>\\n\\n</think>\\n\\n``), as eval and SFT render it."""
    return _render_chat(tokenizer, messages, tools)


def _render_full(tokenizer, messages, tools) -> str:
    messages = _normalize_tool_calls_mv(messages)
    kwargs = dict(tools=tools, tokenize=False, add_generation_prompt=False)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_turn(tokenizer, messages, tools, response) -> str:
    """Assistant text of one decision turn without its ``EOS\\n`` ending (prefix-preserving)."""
    prompt = render(tokenizer, messages, tools)
    full = _render_full(tokenizer, list(messages) + [response], tools)
    ending = tokenizer.eos_token + "\n"
    if not full.startswith(prompt) or not full.endswith(ending) or len(full) == len(prompt) + len(ending):
        raise ValueError("Chat template does not render a prefix-preserving decision turn")
    return full[len(prompt):-len(ending)]


def commit_message(key: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": _draft_and_final(key)}


def call_message(kind: str, key: str, example_id: int, call_id: str) -> Dict[str, Any]:
    """The paper SFT/collection call turn (``marginal_value._tool_call_message``, environment binding)."""
    return _tool_call_message(kind, example_id, key, call_id, BINDING)


def eval_call_message(kind: str, key: str, example_id: int, call_id: str) -> Dict[str, Any]:
    """The same call as the deployed eval keeps it in the history: ``example_id`` appended (stages.py:1785-1786)."""
    message = call_message(kind, key, example_id, call_id)
    function = message["tool_calls"][0]["function"]
    function["arguments"] = json.dumps({**json.loads(function["arguments"]), "example_id": int(example_id)},
                                       ensure_ascii=False)
    return message


def tool_message(kind: str, call_id: str, output: str) -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "name": f"{kind}_tool", "content": output}


def encode_text(tokenizer, text: str) -> List[int]:
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise ValueError(f"Tokenizer cannot round-trip {text!r}")
    if tokenizer.eos_token_id in ids:
        raise ValueError("EOS inside action text")
    return ids


# -------------------------------------------------------------------- actions

@dataclass(frozen=True)
class Action:
    key: str
    kind: Optional[str]  # None = commit
    text: str
    ids: Tuple[int, ...]  # rendered tokens + EOS

    @property
    def name(self) -> str:
        return self.kind or "commit"


def draft_paths(tokenizer, keys: Sequence[str]) -> Dict[str, Tuple[int, ...]]:
    return {k: tuple(encode_text(tokenizer, _draft_only(k))) for k in keys}


def revision_text(draft: str, answer: str) -> str:
    return f"{_draft_only(draft)}\nANSWER_{_answer_token(answer)}"


def revision_paths(tokenizer, keys: Sequence[str]) -> Dict[Tuple[str, str], Tuple[int, ...]]:
    """Forced commit after a revision: ``DRAFT_ANSWER_Y\\nANSWER_Z`` + EOS for every (Y, Z)."""
    drafts = draft_paths(tokenizer, keys)
    paths = {}
    for y in keys:
        for z in keys:
            ids = tuple(encode_text(tokenizer, revision_text(y, z))) + (tokenizer.eos_token_id,)
            if ids[:len(drafts[y])] != drafts[y]:
                raise ValueError(f"revision({y}, {z}) does not start with the DRAFT_ANSWER_{y} tokens")
            paths[(y, z)] = ids
    return paths


def single_token_choice(paths: Dict[Any, Tuple[int, ...]]) -> Optional[Tuple[int, Dict[int, Any]]]:
    """``(position, token -> label)`` when equal-length paths differ in exactly one position, else None."""
    seqs = list(paths.values())
    if len({len(p) for p in seqs}) != 1:
        return None
    diff = [i for i in range(len(seqs[0])) if len({p[i] for p in seqs}) > 1]
    by_token = {p[diff[0]]: label for label, p in paths.items()} if len(diff) == 1 else {}
    return (diff[0], by_token) if by_token and len(by_token) == len(paths) else None


def decision_actions(tokenizer, messages, tools, keys: Sequence[str], kinds: Sequence[str] = ADVISOR_KINDS,
                     example_id: int = 0) -> List[Action]:
    """Every legal turn from this state: commit(K) and call(a, K) for each unused advisor a."""
    drafts = draft_paths(tokenizer, keys)
    actions = []
    for key in keys:
        responses = [(None, commit_message(key))]
        responses += [(kind, call_message(kind, key, example_id, f"grammar_{kind}")) for kind in kinds]
        for kind, response in responses:
            text = render_turn(tokenizer, messages, tools, response)
            ids = tuple(encode_text(tokenizer, text)) + (tokenizer.eos_token_id,)
            if ids[:len(drafts[key])] != drafts[key]:
                raise ValueError(f"{kind or 'commit'}({key}) does not start with the DRAFT_ANSWER_{key} tokens")
            actions.append(Action(key, kind, text, ids))
    return actions


def make_trie(paths) -> ActionTrie:
    paths = [tuple(p) for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate action paths")
    for p in paths:
        if any(q != p and q[:len(p)] == p for q in paths):
            raise ValueError("An action path is a strict prefix of another")
    return ActionTrie(paths)


def complete_path(trie: ActionTrie, prefix) -> Optional[Tuple[int, ...]]:
    prefix = tuple(prefix)
    for path in trie.paths:
        if prefix[:len(path)] == tuple(path):
            return tuple(path)
    return None


def key_token_analysis(tokenizer, keys: Sequence[str], tools=TOOLS_DEPLOY, messages=None) -> Dict[str, Any]:
    """How ``DRAFT_ANSWER_<K>`` and the commit/call continuations tokenise."""
    messages = messages or [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    drafts = draft_paths(tokenizer, keys)
    first = next(iter(drafts.values()))
    n = min(len(d) for d in drafts.values())
    shared = next((i for i in range(n) if len({d[i] for d in drafts.values()}) > 1), n)
    key_tokens = {k: d[shared:] for k, d in drafts.items()}
    single = all(len(t) == 1 for t in key_tokens.values()) and len(set(key_tokens.values())) == len(keys)
    actions = decision_actions(tokenizer, messages, tools, keys)
    diverge, commit_next, call_next = True, set(), set()
    for key in keys:
        commit = next(a for a in actions if a.key == key and a.kind is None)
        for call in (a for a in actions if a.key == key and a.kind):
            i = next(i for i, (x, y) in enumerate(zip(commit.ids, call.ids)) if x != y)
            diverge &= i == len(drafts[key])
            commit_next.add(tokenizer.decode([commit.ids[i]]))
            call_next.add(tokenizer.decode([call.ids[i]]))
    # Manager SFT (routing_anchor.tokenize_anchor_row) supervises tokens(text + EOS + "\n").
    sft_prefix = all(
        list(tokenizer(a.text + tokenizer.eos_token + "\n", add_special_tokens=False)["input_ids"])[:len(a.ids)]
        == list(a.ids) for a in actions)
    return {
        "prefix": [tokenizer.decode([t]) for t in first[:shared]],
        "key_tokens": {k: [tokenizer.decode([t]) for t in v] for k, v in key_tokens.items()},
        "single_token_keys": single,
        "diverge_after_key": diverge,
        "commit_next": sorted(commit_next),
        "call_next": sorted(call_next),
        "sft_target_prefix": sft_prefix,
        "action_lengths": {f"{a.name}({a.key})": len(a.ids) for a in actions},
    }


# -------------------------------------------------------- scoring (pure, torch)

def allowed_logprobs(logits, allowed) -> Dict[int, float]:
    """log-softmax of one logit row renormalised over the allowed token ids."""
    import torch
    allowed = sorted(allowed)
    row = logits.float()[allowed]
    return dict(zip(allowed, (row - torch.logsumexp(row, 0)).tolist()))


def argmax_agreement(step_logits, path) -> Dict[str, Any]:
    """Does unconstrained greedy pick the constrained token at every position?"""
    return path_check([int(step_logits[t].argmax()) for t in range(len(path))], path)


def path_check(argmax: Sequence[int], path, forced: Optional[int] = None) -> Dict[str, Any]:
    """Argmax agreement; the ``forced`` position (the commit forced after a revision) is an
    intervention, reported as ``would_call`` instead of a mismatch."""
    bad = [t for t, token in enumerate(path) if argmax[t] != token and t != forced]
    check = {"checked": len(path) - (forced is not None), "mismatch": len(bad), "first_mismatch": bad[0] if bad else None}
    if forced is not None:
        check["would_call"] = int(argmax[forced] != path[forced])
    return check


def summarize_steps(step_logits, path, trie: ActionTrie) -> Dict[str, Any]:
    """Renormalised path log-prob, per-step allowed distributions, unconstrained argmax and its check."""
    steps = [allowed_logprobs(step_logits[t], trie.allowed(path[:t])) for t in range(len(path))]
    argmax = [int(step_logits[t].argmax()) for t in range(len(path))]
    return {"path": tuple(path), "logprob": sum(s[tok] for s, tok in zip(steps, path)), "steps": steps,
            "argmax": argmax, "check": path_check(argmax, path)}


def merge_checks(checks) -> Dict[str, int]:
    checks = [c for c in checks if c]
    return {"checked": sum(c["checked"] for c in checks), "mismatch": sum(c["mismatch"] for c in checks),
            "would_call": sum(c.get("would_call", 0) for c in checks)}


def constrained_greedy_walk(next_logits, trie: ActionTrie) -> Dict[str, Any]:
    """Reference constrained greedy: ``next_logits(prefix) -> logit row``; one call per position."""
    prefix, rows = [], []
    while complete_path(trie, prefix) is None:
        row = next_logits(prefix)
        rows.append(row)
        prefix.append(max(trie.allowed(prefix), key=lambda t: (float(row[t]), -t)))
    return summarize_steps(rows, prefix, trie)


def score_paths(model, prompt_ids: Sequence[int], paths, trie: ActionTrie, pad_id: int):
    """Exact renormalised log-prob of each path after one shared prompt (one batched forward).

    Right padding is exact for a causal model; gradients flow when enabled by the caller.
    """
    import torch
    device = next(model.parameters()).device
    paths = [list(p) for p in paths]
    width = max(map(len, paths))
    ids = torch.full((len(paths), len(prompt_ids) + width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, p in enumerate(paths):
        seq = list(prompt_ids) + p
        ids[i, :len(seq)] = torch.tensor(seq)
        mask[i, :len(seq)] = 1
    logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), logits_to_keep=width + 1).logits.float()
    out = []
    for i, p in enumerate(paths):
        total = logits.new_zeros(())
        for t, token in enumerate(p):
            row = logits[i, t]
            total = total + row[token] - torch.logsumexp(row[trie.allowed(p[:t])], 0)
        out.append(total)
    return torch.stack(out)


# ------------------------------------------------------------------ HF backend

class HFBackend:
    """Grammar-constrained greedy generation with an HF causal LM (eval's generate call + a prefix constraint).

    ``batch_size`` 1 (default) is eval's batch-1 decoding (``stages.run_eval_manager_tools``
    generates one example at a time). Larger batches left-pad, which shifts
    Qwen3.5's 64-token linear-attention chunks and can flip near-tie keys in
    bf16, so they are a throughput option for non-parity runs only; the batch
    size is part of ``identity`` so a resume never mixes the two.
    """

    def __init__(self, model, tokenizer, device: Optional[str] = None, batch_size: int = 1,
                 identity: Optional[Dict[str, Any]] = None):
        self.model, self.tokenizer = model, tokenizer
        self.device = device or str(next(model.parameters()).device)
        self.batch_size = max(1, int(batch_size))
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self._identity = dict(identity or {})

    @property
    def identity(self) -> Dict[str, Any]:
        """Everything here that can change a decision: weights (caller), template, decoding numerics."""
        import hashlib

        import torch
        device_type = self.device.split(":")[0]
        template = getattr(self.tokenizer, "chat_template", None) or ""
        out = {**self._identity, "batch_size": self.batch_size, "dtype": str(getattr(self.model, "dtype", None)),
               "device_type": device_type,
               "chat_template_sha256": hashlib.sha256(str(template).encode("utf-8")).hexdigest()}
        if device_type == "cuda" and torch.cuda.is_available():
            out.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(self.device))
        return out

    def prompt_ids(self, messages, tools) -> List[int]:
        return list(self.tokenizer(render(self.tokenizer, messages, tools))["input_ids"])

    def greedy(self, prompts: Sequence[Sequence[int]], tries: Sequence[ActionTrie]) -> List[Dict[str, Any]]:
        import torch
        out = []
        for start in range(0, len(prompts), self.batch_size):
            chunk, chunk_tries = prompts[start:start + self.batch_size], tries[start:start + self.batch_size]
            width = max(map(len, chunk))
            ids = torch.tensor([[self.pad_id] * (width - len(p)) + list(p) for p in chunk])
            mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk])

            def allowed(batch_id, input_ids, chunk_tries=chunk_tries):
                trie, prefix = chunk_tries[batch_id], tuple(input_ids[width:].tolist())
                nxt = trie.next.get(prefix)
                if nxt:
                    return sorted(nxt)
                if complete_path(trie, prefix) is not None:
                    return [self.pad_id]
                raise ValueError("Generated prefix is outside the finite action grammar")

            with torch.no_grad():
                gen = self.model.generate(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                    max_new_tokens=max(len(p) for t in chunk_tries for p in t.paths),
                    do_sample=False, pad_token_id=self.pad_id, eos_token_id=self.tokenizer.eos_token_id,
                    prefix_allowed_tokens_fn=allowed, output_logits=True, return_dict_in_generate=True,
                )
            for i, trie in enumerate(chunk_tries):
                path = complete_path(trie, gen.sequences[i, width:].tolist())
                if path is None:
                    raise ValueError("Constrained generation ended before a complete action")
                out.append(summarize_steps([gen.logits[t][i] for t in range(len(path))], path, trie))
        return out


def load_hf_manager(checkpoint: str, base_model: str, revision: Optional[str] = None, batch_size: int = 1,
                    device: Optional[str] = None) -> HFBackend:
    """``stages._load_manager_for_eval``: tokenizer (+ template) from the checkpoint, LoRA on ``base_model``."""
    import os

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    kwargs = {"revision": revision} if revision and not os.path.isdir(base_model) else {}
    if os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, trust_remote_code=True, **kwargs).to(device)
        model = PeftModel.from_pretrained(base, checkpoint).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    identity = {"checkpoint": checkpoint, "base_model": base_model, "base_revision": revision}
    weights = os.path.join(checkpoint, "adapter_model.safetensors")
    if os.path.exists(weights):
        import hashlib
        digest = hashlib.sha256()
        with open(weights, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        identity["adapter_sha256"] = digest.hexdigest()
    return HFBackend(model, tok, device, batch_size, identity=identity)


# --------------------------------------------------------------- decision layer

@dataclass
class Decision:
    key: str
    kind: Optional[str]
    text: str
    logprob: float
    check: Dict[str, Any]
    key_logprobs: Optional[Dict[str, float]] = None
    draft: Optional[str] = None  # revision: the DRAFT key Y (``key`` is the committed ANSWER key Z)
    answer_logprobs: Optional[Dict[str, float]] = None

    @property
    def name(self) -> str:
        return self.kind or "commit"


class Manager:
    """Decision-level manager used by the collector, FA-GRPO and evaluation.

    ``root`` is the grammar-constrained greedy deployment decision; ``revise`` the
    constrained greedy revision after a forced tool output (commit forced, paper
    Eq. 2: ``DRAFT_ANSWER_Y\\nANSWER_Z``, outcome Z); ``probe`` reproduces the
    paper's probe-message generation.
    """

    def __init__(self, backend: HFBackend, tools=TOOLS_DEPLOY, probe_max_new_tokens: int = 512):
        self.backend, self.tools, self.probe_max_new_tokens = backend, tools, probe_max_new_tokens
        self.tokenizer = backend.tokenizer

    def identity(self) -> Dict[str, Any]:
        return {"protocol": PROTOCOL_VERSION, "tools": [t["function"]["description"] for t in self.tools],
                **getattr(self.backend, "identity", {})}

    @staticmethod
    def _choice_logprobs(result, paths) -> Optional[Dict[str, float]]:
        """Renormalised key distribution when the keys' paths differ in one token."""
        choice = single_token_choice(paths)
        if choice is None or set(result["steps"][choice[0]]) != set(choice[1]):
            return None
        pos, by_token = choice
        return {by_token[t]: round(lp, 6) for t, lp in sorted(result["steps"][pos].items())}

    def root(self, messages, keys, kinds, example_id: int) -> Decision:
        actions = decision_actions(self.tokenizer, messages, self.tools, keys, kinds, example_id)
        trie = make_trie(a.ids for a in actions)
        result = self.backend.greedy([self.backend.prompt_ids(messages, self.tools)], [trie])[0]
        action = next(a for a in actions if a.ids == result["path"])
        return Decision(action.key, action.kind, action.text, result["logprob"], result["check"],
                        self._choice_logprobs(result, draft_paths(self.tokenizer, keys)))

    def revise(self, states: Sequence[List[Dict[str, Any]]], keys) -> List[Decision]:
        if not states:
            return []
        keys = list(keys)
        paths = revision_paths(self.tokenizer, keys)
        if render_turn(self.tokenizer, states[0], self.tools, commit_message(keys[0])) != _draft_and_final(keys[0]):
            raise ValueError("Chat template does not render the forced commit verbatim")
        drafts = draft_paths(self.tokenizer, keys)
        by_path = {ids: yz for yz, ids in paths.items()}
        trie = make_trie(paths.values())
        prompts = [self.backend.prompt_ids(m, self.tools) for m in states]
        out = []
        for result in self.backend.greedy(prompts, [trie] * len(prompts)):
            y, z = by_path[result["path"]]
            check = path_check(result["argmax"], result["path"], forced=len(drafts[y]))
            out.append(Decision(z, None, revision_text(y, z), result["logprob"], check,
                                self._choice_logprobs(result, drafts), draft=y,
                                answer_logprobs=self._choice_logprobs(result, {k: paths[(y, k)] for k in keys})))
        return out

    def probe(self, messages, keys, probe: str) -> Tuple[Optional[str], str, bool]:
        return _generate_answer(self.tokenizer, self.backend.model, messages, TOOLS_SFT, list(keys), probe,
                                self.probe_max_new_tokens, 0.0, self.backend.device)


# ------------------------------------------------------------- record parsing

def parse_decision(content: str, tool_call: Optional[Dict[str, Any]], keys) -> Optional[Tuple[str, Optional[str]]]:
    """Map a recorded paper-era turn (eval trajectory event) onto a grammar action, else None."""
    content = str(content or "").strip()
    if tool_call is None:
        for key in keys:
            if content == _draft_and_final(key):
                return key, None
        return None
    name = str(tool_call.get("name") or "")
    kind = name[:-5] if name.endswith("_tool") else name
    args = dict(tool_call.get("arguments") or {})
    args.pop("example_id", None)  # injected by the environment-binding evaluator
    for key in keys:
        if content == _draft_only(key) and kind in ADVISOR_KINDS:
            if args == ({"current_draft": key} if kind == "verifier" else {}):
                return key, kind
    return None
