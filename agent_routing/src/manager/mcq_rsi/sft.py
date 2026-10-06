"""Round-k Manager SFT continuation for MCQ RSI (design §3.4) and the SFT tokenisation parity check (§7.1 item 7).

``train_round_sft`` continues the previous round's adapter G_{k-1} with
``evolve.train_manager_sft`` (same LoRA, new optimizer): 3 epochs, lr 1e-5 with
linear decay, batch 1 x grad-accum 8, ``max_seq_len`` 4096, rendering with the
paper's SFT tool schemas. It validates the label file, checks tokenisation
parity, records a run signature (refusing to resume under different inputs) and
writes ``sft_report.json`` as its completion marker. The adapter lands in
``<out_dir>/model``.

Tokenisation parity. Paper-era Manager SFT (24e6902 ``evolve._tokenize_manager_sft``)
tokenised the full render and masked the common token prefix with the
generation-prompt render (``evolve._mask_prefix_len``). 9.30 tokenises prompt and
target separately (``routing_anchor.build_anchor_features(..., "full")``). Their
supervised token ids agree on all four round-1 label files. Their inputs do not:
the Qwen3.5 patch at the end of ``routing_anchor.py`` normalises *every* list of
dicts passed to ``_render_chat``, including ``tools``, and so adds
``"tool_calls": null`` to each tool schema of the rendered system block, a
context that paper-era SFT, round-1 S_1 and the deployed eval never see.
``context="paper"`` (default) passes the schemas as a tuple, which that patch
leaves alone: inputs and labels are then token-identical to paper-era SFT.
``context="evolve"`` runs ``train_manager_sft`` exactly as 9.30 does.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..marginal_value import ADVISOR_KINDS, _draft_and_final, _draft_only, _tool_schemas
from . import benchmarks as registry
from .collect import _atomic_write

SFT_VERSION = "mcq_rsi_sft/1"
CONTEXTS = ("paper", "evolve")
# Which rows' DRAFT_ANSWER_X tokens are trained. "all": every row (the paper's SFT). "commit_rows": only commit and
# commit_after_call rows (correct drafts); a call row's draft is the manager's own wrong draft, which "all" teaches
# back to the manager round after round (MedQA/AQuA drafts drifted toward one letter, 2026-10-06).
DRAFT_SUPERVISION = ("all", "commit_rows")
DECISION_TYPES = ("commit", "call", "commit_after_call")
_DRAFT_RE = re.compile(r"^DRAFT_ANSWER_([A-Z])")
_PATCH_LOCK = threading.Lock()


@dataclass
class RoundSFTConfig:
    num_train_epochs: int = 3
    learning_rate: float = 1e-5  # linear decay to 0 (TrainingArguments default scheduler), no warmup
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4096
    seed: int = 42
    max_steps: int = -1
    bf16: bool = True
    context: str = "paper"
    draft_supervision: str = "all"

    def validate(self) -> None:
        if self.context not in CONTEXTS:
            raise ValueError(f"context must be one of {CONTEXTS}")
        if self.draft_supervision not in DRAFT_SUPERVISION:
            raise ValueError(f"draft_supervision must be one of {DRAFT_SUPERVISION}")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0 or self.max_seq_len <= 0:
            raise ValueError("epochs, learning rate and max_seq_len must be positive")


def sft_tools(context: str = "paper"):
    """The paper's Manager SFT tool schemas, in the container that renders them as ``context`` says."""
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {CONTEXTS}")
    tools = _tool_schemas("environment")
    return tuple(tools) if context == "paper" else tools


def _read_rows(labels) -> List[Dict[str, Any]]:
    if isinstance(labels, (str, Path)):
        with open(labels, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return list(labels)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ label validation

def _check_call(message: Dict[str, Any], key: str) -> Optional[str]:
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return f"{len(calls)} tool calls"
    function = calls[0].get("function") or {}
    name = str(function.get("name") or "")
    if name not in {f"{k}_tool" for k in ADVISOR_KINDS}:
        return f"unknown tool {name!r}"
    try:
        args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) \
            else dict(function.get("arguments") or {})
    except ValueError:
        return "tool arguments are not JSON"
    expected = {"current_draft": key} if name == "verifier_tool" else {}
    if args != expected:
        return f"{name} arguments {args} != {expected} (environment binding)"
    if str(message.get("content") or "") != _draft_only(key):
        return "call content is not DRAFT_ANSWER_<K>"
    return None


def validate_labels(labels, bench: Optional[registry.Benchmark] = None) -> Dict[str, Any]:
    """Structural checks of a Manager SFT label file (raises ValueError on the first bad row).

    Rows are ``_make_sft_rows`` rows: train split only, ``question_hash`` present,
    (system, user) base prompt (the benchmark's manager system prompt when
    ``bench`` is given), one assistant response of the paper format: commit
    ``DRAFT_ANSWER_K\\nANSWER_K``; call ``DRAFT_ANSWER_K`` + one environment-binding
    advisor call (Verifier ``current_draft=K``); commit after a call ends a
    prompt whose last message is that call's tool output.
    """
    from ...subagents.train import validate_sft_splits
    rows = _read_rows(labels)
    if not rows:
        raise ValueError("empty Manager SFT label file")
    systems = None
    if bench is not None:
        systems = {bench.manager_system_prompt(bench.choice_keys[:n]) for n in range(2, len(bench.choice_keys) + 1)}
    counts = {t: 0 for t in DECISION_TYPES}
    for i, row in enumerate(rows):
        def fail(msg):
            raise ValueError(f"label row {i} (example_id={row.get('example_id')}): {msg}")
        if row.get("split") not in (None, "", "train"):
            fail(f"split {row.get('split')!r} is not train")
        if not isinstance(row.get("example_id"), int) or not str(row.get("question_hash") or ""):
            fail("missing example_id / question_hash")
        kind = row.get("decision_type")
        if kind not in DECISION_TYPES:
            fail(f"decision_type {kind!r}")
        counts[kind] += 1
        prompt, response = row.get("prompt"), row.get("response")
        if not isinstance(prompt, list) or len(prompt) < 2 or [m.get("role") for m in prompt[:2]] != ["system", "user"]:
            fail("prompt must start with (system, user)")
        if systems is not None and prompt[0].get("content") not in systems:
            fail(f"system prompt is not the {bench.name} manager prompt")
        if not isinstance(response, list) or len(response) != 1 or response[0].get("role") != "assistant":
            fail("response must be exactly one assistant message")
        message = response[0]
        match = _DRAFT_RE.match(str(message.get("content") or ""))
        if not match:
            fail("response does not start with DRAFT_ANSWER_<K>")
        key = match.group(1)
        last = prompt[-1].get("role")
        if kind == "call":
            problem = _check_call(message, key)
            if problem:
                fail(problem)
            if last not in ("user", "tool"):
                fail("a call must follow the question or a tool output")
        else:
            if message.get("tool_calls"):
                fail(f"{kind} response carries tool calls")
            if message.get("content") != _draft_and_final(key):
                fail(f"{kind} response is not DRAFT_ANSWER_K\\nANSWER_K")
            if (kind == "commit") != (len(prompt) == 2):
                fail("commit rows have the bare (system, user) prompt; commit_after_call rows follow calls")
            if kind == "commit_after_call" and last != "tool":
                fail("commit_after_call must follow a tool output")
        for m in prompt[2:]:
            if m.get("role") == "assistant" and _check_call(m, str(m.get("content") or "")[-1:]):
                fail(f"history call is malformed: {_check_call(m, str(m.get('content') or '')[-1:])}")
    validate_sft_splits(rows)
    report = {"rows": len(rows), "decision_types": counts,
              "questions": len({(r["example_id"], r["question_hash"]) for r in rows})}
    if isinstance(labels, (str, Path)):
        report.update(path=str(labels), sha256=_sha256(labels))
    return report


# ----------------------------------------------------------------- tokenisation parity

def paper_era_features(rows, tokenizer, max_seq_len: int = 4096, tools=None) -> List[Dict[str, List[int]]]:
    """``evolve._tokenize_manager_sft`` as of 24e6902 (the paper's Manager SFT), without ``datasets``."""
    from ..evolve import _mask_prefix_len, _render_chat
    tools = list(tools) if tools is not None else _tool_schemas("environment")
    eos = tokenizer.eos_token or ""
    out = []
    for ex in rows:
        prompt_msgs, response_msgs = ex["prompt"], ex["response"]
        if isinstance(response_msgs, dict):
            response_msgs = [response_msgs]
        elif isinstance(response_msgs, str):
            response_msgs = [{"role": "assistant", "content": response_msgs}]
        prompt_text = _render_chat(tokenizer, prompt_msgs, add_generation_prompt=True, tools=tools)
        full_text = _render_chat(tokenizer, prompt_msgs + response_msgs, add_generation_prompt=False, tools=tools)
        if eos and not full_text.rstrip().endswith(eos):
            full_text = full_text + eos
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full = tokenizer(full_text, add_special_tokens=False)
        input_ids = full["input_ids"][:max_seq_len]
        plen = min(_mask_prefix_len(prompt_ids, full["input_ids"]), max_seq_len)
        labels = ([-100] * plen) + input_ids[plen:]
        labels = labels[:max_seq_len]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))
        out.append({"input_ids": list(input_ids), "labels": list(labels),
                    "truncated": len(full["input_ids"]) > max_seq_len})
    return out


def tokenization_parity(labels, tokenizer, max_seq_len: int = 4096, context: str = "paper") -> Dict[str, Any]:
    """Compare 9.30 SFT features (``build_anchor_features(..., "full")`` with the ``context`` tool
    container) with paper-era common-prefix masking, row by row.

    ``supervised_mismatch`` counts rows whose supervised token ids differ (the
    §7.1 item 7 gate); ``input_mismatch`` rows whose full input ids differ.
    """
    from ..routing_anchor import build_anchor_features
    rows = _read_rows(labels)
    old = paper_era_features(rows, tokenizer, max_seq_len)
    new, stats = build_anchor_features(rows, tokenizer, max_seq_len, "full", sft_tools(context))
    if len(new) != len(rows):
        raise ValueError(f"9.30 SFT drops rows (empty or overlength targets): {stats}")
    supervised = input_ids = 0
    first = None
    for i, (a, b) in enumerate(zip(old, new)):
        sa = [t for t in a["labels"] if t != -100]
        sb = [t for t in b["labels"] if t != -100]
        bad_sup, bad_in = sa != sb or a["truncated"], a["input_ids"] != b["input_ids"]
        supervised += bad_sup
        input_ids += bad_in
        if (bad_sup or bad_in) and first is None:
            k = next((j for j, (x, y) in enumerate(zip(a["input_ids"], b["input_ids"])) if x != y),
                     min(len(a["input_ids"]), len(b["input_ids"])))
            first = {"row": i, "supervised_equal": not bad_sup, "first_input_difference": k,
                     "paper": tokenizer.decode(a["input_ids"][max(0, k - 8):k + 16]),
                     "ours": tokenizer.decode(b["input_ids"][max(0, k - 8):k + 16])}
    return {"rows": len(rows), "context": context, "supervised_mismatch": supervised, "input_mismatch": input_ids,
            "supervised_tokens": sum(sum(t != -100 for t in f["labels"]) for f in new),
            "max_tokens": max(len(f["input_ids"]) for f in new), "first_difference": first}


def check_tokenization_parity(labels, tokenizer, max_seq_len: int = 4096, context: str = "paper",
                              require_inputs: Optional[bool] = None) -> Dict[str, Any]:
    """Controller/SFT preflight: raise unless supervised ids match (and inputs too for ``paper``)."""
    report = tokenization_parity(labels, tokenizer, max_seq_len, context)
    require_inputs = context == "paper" if require_inputs is None else require_inputs
    if report["supervised_mismatch"] or (require_inputs and report["input_mismatch"]):
        raise ValueError(f"Manager SFT tokenisation differs from paper-era SFT: {report}")
    return report


@contextlib.contextmanager
def sft_context(context: str = "paper", draft_supervision: str = "all"):
    """Run ``evolve.train_manager_sft`` with the tool schemas rendered as ``context`` says and the
    draft tokens trained as ``draft_supervision`` says.

    ``paper`` wraps ``evolve._tokenize_manager_sft`` for the duration so the schemas
    reach ``build_anchor_features`` as a tuple (see module docstring); evolve.py is
    not edited and nothing else changes. ``commit_rows`` tokenises with the
    ``route_only_calls`` anchor mode instead of ``full``: call rows get no loss on their
    ``DRAFT_ANSWER_X`` (``DRAFT_SUPERVISION``).
    """
    if context not in CONTEXTS:
        raise ValueError(f"context must be one of {CONTEXTS}")
    if draft_supervision not in DRAFT_SUPERVISION:
        raise ValueError(f"draft_supervision must be one of {DRAFT_SUPERVISION}")
    if context == "evolve" and draft_supervision == "all":
        yield
        return
    from .. import evolve
    with _PATCH_LOCK:
        original = evolve._tokenize_manager_sft

        def tokenize(rows, tok, max_seq_len, tools=None):
            tools = tuple(tools) if context == "paper" and tools else tools
            if draft_supervision == "all":
                return original(rows, tok, max_seq_len, tools=tools)
            from datasets import Dataset
            from ..routing_anchor import build_anchor_features
            features, stats = build_anchor_features(rows, tok, max_seq_len, "route_only_calls", tools)
            if len(features) != len(rows):
                raise ValueError(f"Manager SFT contains empty or overlength targets; no silent truncation: {stats}")
            return Dataset.from_list(features)

        evolve._tokenize_manager_sft = tokenize
        try:
            yield
        finally:
            evolve._tokenize_manager_sft = original


# ------------------------------------------------------------------------- training

def _lora(path) -> Dict[str, Any]:
    from peft import PeftConfig
    cfg = PeftConfig.from_pretrained(str(path))
    return {"r": cfg.r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
            "target_modules": sorted(cfg.target_modules), "base_model_name_or_path": cfg.base_model_name_or_path}


def source_sha256() -> str:
    """Code that shapes round SFT beyond ``harness_identity``: this wrapper and the rest of the package
    (label checks, protocol, prompts, select), ``train_manager_sft`` (``evolve.py``), the renderer
    (``routing_anchor.py``), ``manager/prompt.py``, ``stages.py`` and ``subagents/train.py``
    (``grpo.source_files``)."""
    from .grpo import files_sha256, source_files
    return files_sha256(source_files())


def train_round_sft(labels, init_adapter, out_dir, *, base_model: str = registry.BASE_MODEL,
                    base_revision: Optional[str] = registry.BASE_REVISION,
                    bench: Optional[str] = None, config=None) -> Dict[str, Any]:
    """SFT_k: continue ``init_adapter`` (G_{k-1}) on round-k ``labels``; returns ``sft_report.json``.

    The base is pinned: a hub id is replaced by its local snapshot at ``base_revision``
    (``evaluate.resolve_base``; a local directory is used as given), so ``train_manager_sft``
    loads the commit FA-GRPO and the eval load. ``base_revision`` None / "" leaves it unpinned.
    Both, and the resolved base's identity, are part of the run signature.
    """
    from transformers import AutoTokenizer

    from ...verifiable.provenance import harness_identity
    from ...verifiable.runner import checkpoint_identity
    from ..evolve import ManagerSFTConfig, train_manager_sft
    from . import evaluate
    cfg = config if isinstance(config, RoundSFTConfig) else RoundSFTConfig(**dict(config or {}))
    cfg.validate()
    base_revision = base_revision or None
    base_model = evaluate.resolve_base(base_model, base_revision)
    init_adapter, root, labels = Path(init_adapter), Path(out_dir), Path(labels)
    if not (init_adapter / "adapter_config.json").is_file():
        raise ValueError(f"{init_adapter}: round-k SFT continues a LoRA adapter (G_(k-1))")
    if not (init_adapter / "tokenizer_config.json").is_file():
        raise ValueError(f"{init_adapter}: adapter has no tokenizer (train_manager_sft loads it from the init)")
    spec = registry.get(bench) if bench else None
    label_report = validate_labels(labels, spec)
    tokenizer = AutoTokenizer.from_pretrained(str(init_adapter), trust_remote_code=True)
    parity = check_tokenization_parity(labels, tokenizer, cfg.max_seq_len, cfg.context)
    lora = _lora(init_adapter)
    signature = json.loads(json.dumps({
        "version": SFT_VERSION, "config": asdict(cfg), "base_model": base_model, "base_revision": base_revision,
        "bench": bench,
        "labels": str(labels), "labels_sha256": label_report["sha256"], "init_adapter": str(init_adapter),
        "init_identity": checkpoint_identity(init_adapter), "lora": lora, "harness": harness_identity(),
        "source_sha256": source_sha256(), "base_identity": evaluate.base_identity(base_model, base_revision),
        "scheduler": "linear", "warmup_steps": 0, "optimizer": "new AdamW (TrainingArguments default)",
    }))
    root.mkdir(parents=True, exist_ok=True)
    sig_path, report_path = root / "run_signature.json", root / "sft_report.json"
    if sig_path.exists():
        old = json.loads(sig_path.read_text(encoding="utf-8"))
        if old != signature:
            diff = sorted(k for k in set(old) | set(signature) if old.get(k) != signature.get(k))
            raise ValueError(f"{root}: round SFT inputs changed ({diff}); use a new output directory")
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
    elif any(root.iterdir()):
        raise ValueError(f"{root}: non-empty round SFT output without run_signature.json")
    else:
        _atomic_write(sig_path, json.dumps(signature, indent=2, sort_keys=True) + "\n")
    model_dir = root / "model"
    manager_cfg = ManagerSFTConfig(
        base_model=base_model, train_jsonl=str(labels), out_dir=str(model_dir),
        init_model_or_adapter=str(init_adapter), seed=cfg.seed, max_seq_len=cfg.max_seq_len,
        learning_rate=cfg.learning_rate, num_train_epochs=cfg.num_train_epochs,
        per_device_batch_size=cfg.per_device_batch_size, gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        use_lora=True, lora_r=lora["r"], lora_alpha=lora["lora_alpha"], lora_dropout=lora["lora_dropout"],
        max_steps=cfg.max_steps, bf16=cfg.bf16)
    with sft_context(cfg.context, cfg.draft_supervision):
        train_manager_sft(manager_cfg)
    if not (model_dir / "adapter_config.json").is_file():
        raise RuntimeError(f"{model_dir}: train_manager_sft wrote no adapter")
    trained = _lora(model_dir)
    if {k: trained[k] for k in ("r", "lora_alpha", "target_modules")} != {k: lora[k] for k in ("r", "lora_alpha", "target_modules")}:
        raise RuntimeError(f"continued adapter changed its LoRA config: {lora} -> {trained}")
    metrics_path = model_dir / "training_metrics.json"
    report = {
        "version": SFT_VERSION, "model_dir": str(model_dir), "init_adapter": str(init_adapter),
        "labels": label_report, "tokenization_parity": parity, "lora": trained, "config": asdict(cfg),
        "adapter_sha256": _sha256(model_dir / "adapter_model.safetensors"),
        "training_metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None,
        "signature_sha256": hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest(),
    }
    _atomic_write(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
