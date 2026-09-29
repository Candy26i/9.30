"""Text model loading and a frozen advisor endpoint; imports are lazy for CPU tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .protocol import advisor_messages
from .telemetry import generation, usage, progress
from .sampling import normalize_generation, generation_kwargs


class ContextBudgetExceeded(ValueError):
    """This prompt cannot be generated under the declared per-sample budget."""
    def __init__(self, prompt_tokens, max_tokens, max_context):
        self.prompt_tokens, self.max_tokens, self.max_context = prompt_tokens, max_tokens, max_context
        super().__init__(f"Context budget exceeded: {prompt_tokens} + {max_tokens} > {max_context}; no silent truncation")


def configure_tokenizer(tok):
    """Use one explicit Qwen ChatML/JSON-tool protocol in every experiment phase.

    TRL 0.29 only provides Qwen3 JSON response parsing. Qwen3.5's default XML
    template is therefore NOT silently mixed with the GRPO parser. This fixed
    template is part of the experimental harness, including baseline evaluation.
    """
    from copy import deepcopy
    from trl.chat_template_utils import qwen3_schema
    if "<|im_start|>" not in tok.get_vocab() or tok.eos_token != "<|im_end|>":
        raise ValueError("This math runner currently supports Qwen ChatML tokenizers only")
    tok.chat_template = Path(__file__).with_name("chat_template.jinja").read_text()
    tok.response_schema = deepcopy(qwen3_schema)
    return tok


def load_model(base_model, checkpoint=None, trainable=False, lora_rank=16, revision=None):
    progress(phase="loading_model", checkpoint=checkpoint or base_model)
    import torch
    import transformers as tr
    from peft import LoraConfig, PeftModel, get_peft_model

    source = checkpoint or base_model
    adapter = Path(source).is_dir() and (Path(source) / "adapter_config.json").exists()
    weights = base_model if adapter else source
    if adapter:
        saved = json.loads((Path(source) / "adapter_config.json").read_text())
        recorded = saved.get("base_model_name_or_path")
        if recorded and recorded != base_model:
            raise ValueError(f"Adapter base {recorded!r} differs from configured {base_model!r}")
    revision_args = {"revision": revision} if revision and not Path(weights).is_dir() else {}
    config = tr.AutoConfig.from_pretrained(weights, **revision_args)
    # Qwen3.5 official checkpoints contain a multimodal config. Use its text
    # causal LM class (documented by Transformers), not AutoModelForCausalLM's
    # mapping for the enclosing multimodal config.
    cls = (getattr(tr, "Qwen3_5ForCausalLM") if config.model_type == "qwen3_5"
           else tr.AutoModelForCausalLM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cls.from_pretrained(weights, dtype=torch.bfloat16 if device == "cuda" else torch.float32, **revision_args)
    tokenizer_args = revision_args if not adapter else {}
    tok = configure_tokenizer(tr.AutoTokenizer.from_pretrained(source, **tokenizer_args))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    if adapter:
        model = PeftModel.from_pretrained(model, source, is_trainable=trainable)
    elif trainable:
        candidates = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                      "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"}
        present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
        targets = sorted(candidates & present)
        if not targets:
            raise ValueError("No supported LoRA projection modules in this model")
        model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=lora_rank * 2,
                               lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                               target_modules=targets))
    model.to(device)
    if trainable:
        model.config.use_cache = False
        model.enable_input_require_grads()
    else:
        model.eval()
        model.requires_grad_(False)
    usage("model_load", {}, base_model=base_model, checkpoint=source,
          resolved_revision=getattr(config, "_commit_hash", None),
          template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest())
    return tok, model


def render(tokenizer, messages, tools=None, generation=True):
    return tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
           add_generation_prompt=generation, enable_thinking=False)


def strip_generation_endings(text, tokenizer):
    """Remove terminal EOS/padding only; preserve tool markers and answer text.

    There may be multiple endings (for example ChatML EOS followed by padding).
    A single pass over a set makes cleanup depend on iteration order and leaves
    EOS behind when EOS and padding are the same token and occur repeatedly.
    """
    endings = sorted({t for t in (tokenizer.eos_token, tokenizer.pad_token) if t},
                     key=lambda t: (-len(t), t))
    text = text.strip()
    while True:
        for token in endings:
            if text.endswith(token):
                text = text[:-len(token)].rstrip()
                break
        else:
            return text


class HFBackend:
    usage_actor = "manager"  # Also inherited by rollout backends with their own initializer.

    def __init__(self, base_model, checkpoint=None, max_context=16384, revision=None, decision_constraint="none",
                 usage_actor="manager"):
        if type(max_context) is not int or max_context <= 0:
            raise ValueError("max_context must be a positive integer")
        self.tokenizer, self.model = load_model(base_model, checkpoint, revision=revision)
        self.max_context = max_context
        self.usage_actor = usage_actor
        if decision_constraint not in {"none", "finite_actions_v1"}:
            raise ValueError("Unknown decision constraint")
        self.decision_constraint = decision_constraint

    def generate(self, messages, tools=None, max_tokens=2048, temperature=0.0, seed=42,
                 generation_options=None):
        import torch
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        settings = normalize_generation({"temperature": temperature, "seed": seed,
                                         **(generation_options or {})})
        prompt = render(self.tokenizer, messages, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        n = inputs["input_ids"].shape[1]
        if n + max_tokens > self.max_context:
            raise ContextBudgetExceeded(n, max_tokens, self.max_context)
        devices = [self.model.device.index or 0] if self.model.device.type == "cuda" else []
        start = time.monotonic()
        grammar = {}
        paths = None
        if tools and getattr(self, "decision_constraint", "none") == "finite_actions_v1":
            from .actions import ActionTrie, decision_paths
            paths = decision_paths(self.tokenizer, messages, tools, max_tokens)
            grammar = ActionTrie(paths).generation_kwargs(n)
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(settings["seed"])
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs(settings, n), **grammar)
        ids = out[0, n:]
        if paths is not None and ids.tolist() not in paths:
            raise ValueError("Constrained decision did not finish a legal action")
        # Stop on the same ChatML EOS used by the tokenizer/template, rather
        # than a potentially different model-level generation default.
        text = strip_generation_endings(
            self.tokenizer.decode(ids, skip_special_tokens=False), self.tokenizer)
        truncated = bool(len(ids) >= max_tokens and int(ids[-1]) != self.tokenizer.eos_token_id)
        result = {"text": text,
                "prompt_tokens": n, "completion_tokens": len(ids),
                "seconds": time.monotonic() - start,
                "truncated": truncated, "finish_reason": "length" if truncated else "stop"}
        usage(getattr(self, "usage_actor", "manager"), result)
        return result


class HTTPAdvisors:
    """Frozen endpoint with an explicit, finite generation budget.

    Transport/schema/identity failures raise. Budget exhaustion and empty model
    completions are observed outcomes, retained with diagnostics rather than
    crashing an entire benchmark. A partial hint is never itself a math label.
    The complete candidate is included in cache identities.
    """
    def __init__(self, url, max_tokens=1024, models=None, timeout=600, generation_options=None):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("advisor max_tokens must be a positive integer")
        self.url = url.rstrip("/")
        self.max_tokens = max_tokens
        self.models = models or {k: k for k in ("extractor", "reasoner", "verifier")}
        self.timeout = timeout
        self.cache = {}
        self.identity = None
        self.generation_options = normalize_generation(generation_options)
        self.require_generation_echo = generation_options is not None

    def call(self, kind, row, draft=""):
        import requests
        msgs = advisor_messages(kind, row, draft)
        body = {"model": self.models[kind], "messages": msgs, **self.generation_options,
                "max_tokens": self.max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            value = dict(self.cache[key])
            value.update(cache_hit=True, actual_completion_tokens=0, actual_prompt_tokens=0, seconds=0.)
            usage("advisor", value, advisor=kind)
            generation("advisor", value, messages=msgs, advisor=kind, max_tokens=self.max_tokens,
                       operation="advice", error=value.get("error"))
            return value
        start = time.monotonic()
        response = requests.post(self.url + "/v1/chat/completions", json=body, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if (self.require_generation_echo
                and data.get("margent_generation") != self.generation_options):
            raise RuntimeError("Advisor did not confirm requested generation settings; restart the updated server")
        fingerprint = data.get("margent_advisor")
        if self.identity is not None and fingerprint != self.identity:
            raise RuntimeError("Advisor identity changed during the stage")
        role_identity = data.get("margent_role")
        if isinstance(fingerprint, dict) and "expert_bundle" in fingerprint:
            from .serve import EXPERT_ADAPTERS, expert_bundle_sha256
            bundle = fingerprint["expert_bundle"]
            if (not isinstance(bundle, dict) or not isinstance(bundle.get("roles"), dict)
                    or set(bundle["roles"]) != set(EXPERT_ADAPTERS)
                    or any(not isinstance(entry, dict) for entry in bundle["roles"].values())
                    or fingerprint.get("expert_bundle_sha256") != expert_bundle_sha256(bundle)):
                raise RuntimeError("Advisor expert bundle fingerprint is malformed")
            expected_role = {"role": kind, "adapter_name": EXPERT_ADAPTERS[kind],
                             "identity": bundle["roles"][kind].get("identity")}
            if (role_identity != expected_role or data.get("actual_role") != kind
                    or not isinstance(expected_role["identity"], dict) or not expected_role["identity"]):
                raise RuntimeError("Advisor returned a different or unverified expert role/adapter")
        if fingerprint is not None:
            self.identity = fingerprint
        try:
            choices, counts = data["choices"], data["usage"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("exactly one choice required")
            choice = choices[0]
            text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            if not isinstance(text, str):
                raise ValueError("content must be a string")
            # Missing finish reasons are accepted for legacy OpenAI-compatible
            # servers, with conservative length detection from token usage.
            if finish_reason not in {None, "stop", "length"}:
                raise ValueError(f"unexpected finish_reason: {finish_reason}")
            for name in ("prompt_tokens", "completion_tokens"):
                if type(counts[name]) is not int or counts[name] < 0:
                    raise ValueError(f"{name} must be a nonnegative integer")
            if counts["completion_tokens"] > self.max_tokens:
                raise ValueError("completion exceeds requested token budget")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Malformed advisor response: {exc}") from exc
        budget_error = data.get("margent_generation_error")
        if budget_error not in {None, "context_budget_exceeded"}:
            raise RuntimeError(f"Unknown advisor generation error: {budget_error}")
        if budget_error and (text or counts["completion_tokens"]):
            raise RuntimeError("Malformed advisor context-budget failure")
        truncated = finish_reason == "length" or (finish_reason is None and counts["completion_tokens"] >= self.max_tokens)
        error = ("advisor_context_budget_exceeded" if budget_error else
                 "advisor_output_truncated" if truncated else "advisor_empty_output" if not text.strip() else None)
        result = {"text": text, "prompt_tokens": counts["prompt_tokens"],
                  "completion_tokens": counts["completion_tokens"],
                  "actual_prompt_tokens": 0 if budget_error else counts["prompt_tokens"],
                  "actual_completion_tokens": counts["completion_tokens"],
                  "seconds": time.monotonic() - start, "cache_hit": False,
                  "truncated": truncated, "finish_reason": finish_reason,
                  "valid_output": bool(text.strip()) and not truncated, "error": error}
        if role_identity is not None:
            result.update(advisor_role=role_identity, actual_role=data.get("actual_role"))
        if budget_error:
            result["max_context"] = data.get("margent_max_context")
        usage("advisor", result, advisor=kind, advisor_identity=fingerprint,
              advisor_generation=self.generation_options)
        generation("advisor", result, messages=msgs, advisor=kind, max_tokens=self.max_tokens,
                   operation="advice", error=error)
        self.cache[key] = result
        return result
